// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { cloneDeep, range } from 'lodash';

import {
    ActionParameterType, BaseCollectionAction, Job, MLModel, ObjectState,
    ObjectType, ShapeType, Source, Task, getCore, type MinimalShape, type TrackerResults,
} from 'cvat-core-wrapper';

const core = getCore();

export const SAM2_TRACKER_MODEL_ID = 'pth-facebookresearch-sam2';

type Collection = Parameters<BaseCollectionAction['run']>[0]['collection'];
type Shape = Collection['shapes'][number];
type Track = Collection['tracks'][number];
type TrackShape = Track['shapes'][number];
type SAM2TrackerResults = Omit<TrackerResults, 'shapes'> & { shapes: (MinimalShape | null)[] };

const noChanges = {
    created: { shapes: [], tags: [], tracks: [] },
    deleted: { shapes: [], tags: [], tracks: [] },
};

function getTrackShape(track: Track, frame: number): TrackShape | null {
    const shapes = track.shapes
        .filter((shape) => shape.frame <= frame)
        .sort((a, b) => a.frame - b.frame);

    return shapes[shapes.length - 1] ?? null;
}

export default class SAM2TrackerAction extends BaseCollectionAction {
    #instance: Job | Task | null = null;
    #targetFrame = 0;
    #convertPolygonShapesToTracks = false;
    readonly #model: MLModel;

    public constructor(model: MLModel) {
        super();
        this.#model = model;
    }

    public async init(instance: Job | Task, parameters: Record<string, string | number>): Promise<void> {
        this.#instance = instance;
        this.#targetFrame = +parameters['Target frame'];
        this.#convertPolygonShapesToTracks = parameters['Convert polygon shapes to tracks'] === 'true';
    }

    public async destroy(): Promise<void> {
        this.#instance = null;
    }

    public async run({
        collection, frameData: { number }, onProgress, cancelled,
    }: Parameters<BaseCollectionAction['run']>[0]): ReturnType<BaseCollectionAction['run']> {
        if (this.#instance === null || number === this.#targetFrame) {
            return noChanges;
        }

        if (number > this.#targetFrame) {
            throw new Error('SAM2 tracking backward is not supported');
        }

        const frameNumbers = this.#instance instanceof Job ?
            await this.#instance.frames.frameNumbers() : range(0, this.#instance.size);
        const targetFrames = frameNumbers
            .filter((frame) => frame > number && frame <= this.#targetFrame)
            .sort((left, right) => left - right);
        if (!targetFrames.length) {
            return noChanges;
        }

        // Tracks can be interpolated on the current frame. Use the ObjectState rather
        // than the preceding serialized keyframe so SAM2 receives the visible geometry.
        const objectStates = await this.#instance.annotations.get(number, false, []);
        const tracks = cloneDeep(collection.tracks);
        const [initialShapes, targetObjects, targetObjectStates] =
            ([].concat(collection.shapes, tracks) as (Shape | Track)[]).reduce((acc, object) => {
                if (!Number.isInteger(object.clientID)) {
                    return acc;
                }

                const objectState = objectStates.find((state) => state.clientID === object.clientID);
                if (!objectState) {
                    return acc;
                }

                acc[0].push({ type: objectState.shapeType, points: [...objectState.points as number[]] });
                if (
                    this.#convertPolygonShapesToTracks &&
                    objectState.objectType === ObjectType.SHAPE &&
                    objectState.shapeType === ShapeType.POLYGON
                ) {
                    const shape = object as Required<Shape>;
                    const track: Track = {
                        source: Source.AUTO,
                        attributes: [],
                        elements: [],
                        frame: shape.frame,
                        group: shape.group,
                        label_id: shape.label_id,
                        shapes: [{
                            attributes: [],
                            frame: shape.frame,
                            occluded: shape.occluded,
                            outside: false,
                            points: [...objectState.points as number[]],
                            rotation: shape.rotation,
                            z_order: shape.z_order,
                            type: shape.type,
                        }],
                    };
                    tracks.push(track);
                    acc[1].push(track);
                    acc[2].push(new Proxy(objectState, {
                        get(state, property, receiver) {
                            if (property === 'objectType') {
                                return ObjectType.TRACK;
                            }

                            return Reflect.get(state, property, receiver);
                        },
                    }));
                } else {
                    acc[1].push(object);
                    acc[2].push(objectState);
                }

                return acc;
            }, [[], [], []] as [MinimalShape[], (Shape | Track)[], ObjectState[]]);

        if (!initialShapes.length) {
            return noChanges;
        }

        const taskID = this.#instance instanceof Job ? this.#instance.taskId : this.#instance.id;
        if (taskID === null) {
            throw new Error('SAM2 tracker requires a task ID');
        }
        const job = this.#instance instanceof Job ? { job: this.#instance.id } : {};

        onProgress('Initializing SAM2 tracker', 0);
        if (cancelled()) {
            return noChanges;
        }
        const initialized = await core.lambda.call(taskID, this.#model, {
            type: 'init_tracking',
            frame: number,
            ...job,
            shapes: initialShapes,
        }) as SAM2TrackerResults;

        if (!Array.isArray(initialized.states) || initialized.states.length !== initialShapes.length) {
            throw new Error('SAM2 tracker returned an invalid initialization response');
        }
        if (cancelled()) {
            return noChanges;
        }

        let states = initialized.states;
        const trackedShapes: Shape[] = [];
        let hasTrackedFrame = false;
        for (let index = 0; index < targetFrames.length; index++) {
            if (cancelled()) {
                return noChanges;
            }

            const frame = targetFrames[index];
            const frameData = await this.#instance.frames.get(frame);
            if (frameData.deleted) {
                continue;
            }
            if (cancelled()) {
                return noChanges;
            }

            hasTrackedFrame = true;
            onProgress('Tracking with SAM2', Math.round((index / targetFrames.length) * 100));
            const result = await core.lambda.call(taskID, this.#model, {
                type: 'track',
                frame,
                ...job,
                states,
            }) as SAM2TrackerResults;
            if (
                !Array.isArray(result.states) ||
                !Array.isArray(result.shapes) ||
                result.states.length !== initialShapes.length ||
                result.shapes.length !== initialShapes.length
            ) {
                throw new Error('SAM2 tracker returned an invalid tracking response');
            }
            states = result.states;
            if (cancelled()) {
                return noChanges;
            }

            for (let targetIndex = 0; targetIndex < targetObjects.length; targetIndex++) {
                const targetObject = targetObjects[targetIndex];
                const targetObjectState = targetObjectStates[targetIndex];
                const prediction = result.shapes[targetIndex] ?? null;
                if (targetObjectState.objectType === ObjectType.TRACK) {
                    const track = targetObject as Track;
                    const existing = track.shapes.find((shape) => shape.frame === frame);
                    const updated = {
                        attributes: cloneDeep(existing?.attributes ?? []),
                        frame,
                        occluded: existing?.occluded ?? targetObjectState.occluded,
                        outside: prediction === null,
                        points: prediction?.points ?? existing?.points ?? [...targetObjectState.points as number[]],
                        rotation: existing?.rotation ?? 0,
                        z_order: existing?.z_order ?? targetObjectState.zOrder,
                        type: targetObjectState.shapeType,
                    };

                    if (existing) {
                        Object.assign(existing, updated);
                    } else {
                        track.shapes.push(updated);
                    }
                } else if (prediction) {
                    const shape = targetObject as Shape;
                    trackedShapes.push({
                        elements: cloneDeep(shape.elements),
                        group: shape.group,
                        attributes: cloneDeep(shape.attributes),
                        frame,
                        label_id: shape.label_id,
                        points: [...prediction.points],
                        source: Source.AUTO,
                        type: prediction.type,
                        rotation: shape.rotation,
                        outside: false,
                        occluded: shape.occluded,
                        z_order: shape.z_order,
                    });
                }
            }
        }

        if (!hasTrackedFrame) {
            return noChanges;
        }

        onProgress('Tracking with SAM2', 100);
        return {
            created: { shapes: trackedShapes, tags: [], tracks },
            deleted: {
                shapes: this.#convertPolygonShapesToTracks ?
                    collection.shapes.filter((shape) => shape.type === ShapeType.POLYGON) : [],
                tags: [],
                tracks: collection.tracks,
            },
        };
    }

    public applyFilter(
        input: Parameters<BaseCollectionAction['applyFilter']>[0],
    ): ReturnType<BaseCollectionAction['applyFilter']> {
        const { collection, frameData } = input;
        return {
            shapes: collection.shapes.filter((shape) => (
                shape.frame === frameData.number && [ShapeType.MASK, ShapeType.POLYGON].includes(shape.type)
            )),
            tags: [],
            tracks: collection.tracks.filter((track) => {
                const shape = getTrackShape(track, frameData.number);
                return shape?.type === ShapeType.POLYGON && !shape.outside;
            }),
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        return [ShapeType.MASK, ShapeType.POLYGON].includes(objectState.shapeType);
    }

    public get name(): string {
        return 'Segment Anything 2: Tracker';
    }

    public get parameters(): BaseCollectionAction['parameters'] {
        return {
            'Convert polygon shapes to tracks': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: String(this.#convertPolygonShapesToTracks),
            },
            'Target frame': {
                type: ActionParameterType.NUMBER,
                values: ({ instance }) => (instance instanceof Job ?
                    [instance.startFrame, instance.stopFrame, 1] : [0, instance.size - 1, 1]
                ).map((value) => value.toString()),
                defaultValue: ({ instance }) => (instance instanceof Job ?
                    instance.stopFrame : instance.size - 1
                ).toString(),
            },
        };
    }
}
