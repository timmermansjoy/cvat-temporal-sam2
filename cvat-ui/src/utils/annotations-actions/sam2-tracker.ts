// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { cloneDeep, isEqual, range } from 'lodash';

import { EventScope } from 'cvat-logger';
import {
    ActionParameterType, BaseCollectionAction, Job, MLModel, ObjectState,
    ObjectType, ShapeType, Source, Task, getCore, type MinimalShape, type TrackerResults,
} from 'cvat-core-wrapper';

const core = getCore();

export const SAM2_TRACKER_MODEL_ID = 'pth-facebookresearch-sam2';
export const SAM2_TRACKER_ACTION_NAME = 'Segment Anything 2: Tracker';

type Collection = Parameters<BaseCollectionAction['run']>[0]['collection'];
type Shape = Collection['shapes'][number];
type Track = Collection['tracks'][number];
type TrackShape = Track['shapes'][number];
type SAM2ServerTiming = {
    decode_ms?: number;
    preprocess_cpu_ms?: number;
    state_load_ms?: number;
    encoder_gpu_ms?: number;
    tracker_wall_ms?: number;
    tracker_gpu_ms?: number;
    state_save_ms?: number;
    state_bytes?: number;
    server_total_ms?: number;
};
type SAM2TrackerResults = Omit<TrackerResults, 'shapes'> & {
    shapes: (MinimalShape | null)[];
    device?: string;
    model_id?: string;
    timing?: SAM2ServerTiming;
};
type Direction = -1 | 1;
type SessionShape = MinimalShape & { clientID: number; groupID: number; labelID: number };
type TrackingSession = {
    frame: number;
    shapes: SessionShape[];
    states: TrackerResults['states'];
    contextFrames: number;
    anchorFrame: number;
};
type InferenceContext = { frames: number; anchor: number; reused: boolean };
type PredictionWindow = {
    jobKey: string;
    objects: Pick<SessionShape, 'clientID'>[];
    predictions: Map<number, SessionShape[]>;
};

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
    #frameCount = 5;
    #convertPolygonShapesToTracks = false;
    readonly #model: MLModel;
    // ponytail: one active object set per job/direction; key by committed object IDs when the action API exposes them.
    readonly #sessions = new Map<string, TrackingSession>();
    readonly #predictionWindows: PredictionWindow[] = [];

    #logInference(
        operation: 'initialize' | 'track' | 'correct',
        frame: number,
        startedAt: number,
        objectCount: number,
        objectKey: number,
        runID: string,
        direction: Direction,
        result: SAM2TrackerResults,
        context: InferenceContext,
    ): void {
        if (!this.#instance) {
            return;
        }

        this.#instance.logger.log(EventScope.sam2Inference, {
            ...(result.timing || {}),
            duration: Math.round(performance.now() - startedAt),
            operation,
            frame,
            object_count: objectCount,
            object_key: objectKey,
            run_id: runID,
            direction: direction === 1 ? 'forward' : 'backward',
            device: result.device || 'unknown',
            model_id: result.model_id || String(this.#model.id),
            model_name: this.#model.name,
            model_version: this.#model.version,
            context_frames: context.frames,
            distance_from_anchor: Math.abs(frame - context.anchor),
            session_reused: context.reused,
            video_name: this.#instance instanceof Job ?
                this.#instance.taskName || `Task ${this.#instance.taskId}` : this.#instance.name,
        });
    }

    public constructor(model: MLModel) {
        super();
        this.#model = model;
    }

    public async init(instance: Job | Task, parameters: Record<string, string | number>): Promise<void> {
        this.#instance = instance;
        this.#targetFrame = +parameters['Target frame'];
        this.#frameCount = +parameters['Frame count'];
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

        const direction: Direction = this.#targetFrame > number ? 1 : -1;
        const runID = `${Date.now()}-${number}-${direction}`;

        const frameNumbers = this.#instance instanceof Job ?
            await this.#instance.frames.frameNumbers() : range(0, this.#instance.size);
        const targetFrames = frameNumbers
            .filter((frame) => (direction === 1 ?
                frame > number && frame <= this.#targetFrame :
                frame < number && frame >= this.#targetFrame))
            .sort((left, right) => direction * (left - right));
        if (!targetFrames.length) {
            return noChanges;
        }

        // Tracks can be interpolated on the current frame. Use the ObjectState rather
        // than the preceding serialized keyframe so SAM2 receives the visible geometry.
        const objectStates = await this.#instance.annotations.get(number, false, []);
        const taskCollection = this.#instance instanceof Task ?
            await this.#instance.annotations.export() : null;
        let nextGroupID = taskCollection ? Math.max(
            0,
            ...taskCollection.shapes.map((shape) => shape.group ?? 0),
            ...taskCollection.tracks.map((track) => track.group ?? 0),
        ) : 0;
        if (taskCollection) {
            for (const object of ([] as (Shape | Track)[]).concat(collection.shapes, collection.tracks)) {
                if (!object.group) {
                    object.group = ++nextGroupID;
                }
            }
        }
        const tracks = cloneDeep(collection.tracks);
        const [initialShapes, targetObjects, targetObjectStates, sessionShapes] =
            ([].concat(collection.shapes, tracks) as (Shape | Track)[]).reduce((acc, object) => {
                if (!Number.isInteger(object.clientID)) {
                    return acc;
                }

                const objectState = objectStates.find((state) => state.clientID === object.clientID);
                if (!objectState) {
                    return acc;
                }

                acc[0].push({ type: objectState.shapeType, points: [...objectState.points as number[]] });
                acc[3].push({
                    type: objectState.shapeType,
                    points: [...objectState.points as number[]],
                    clientID: object.clientID,
                    groupID: object.group,
                    labelID: object.label_id,
                });
                if (
                    objectState.objectType === ObjectType.SHAPE &&
                    (
                        objectState.shapeType === ShapeType.MASK ||
                        (this.#convertPolygonShapesToTracks && objectState.shapeType === ShapeType.POLYGON)
                    )
                ) {
                    const shape = object as Required<Shape>;
                    const track: Track = {
                        clientID: shape.clientID,
                        source: shape.source,
                        attributes: [],
                        elements: [],
                        frame: shape.frame,
                        group: shape.group,
                        label_id: shape.label_id,
                        shapes: [{
                            attributes: [],
                            frame: shape.frame,
                            source: shape.source,
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
            }, [[], [], [], []] as [MinimalShape[], (Shape | Track)[], ObjectState[], SessionShape[]]);

        if (!initialShapes.length) {
            return noChanges;
        }

        const linkedTracks: Track[] = [];
        if (taskCollection) {
            for (const targetObject of targetObjects) {
                if (!('shapes' in targetObject)) {
                    continue;
                }
                const targetTrack = targetObject as Track;
                const linked = taskCollection.tracks.filter((track) => (
                    track.id !== targetTrack.id &&
                    track.group === targetTrack.group &&
                    track.label_id === targetTrack.label_id &&
                    track.shapes.every((shape) => shape.type === targetTrack.shapes[0].type)
                ));
                linkedTracks.push(...linked);
                const shapesByFrame = new Map(targetTrack.shapes.map((shape) => [shape.frame, shape]));
                for (const track of linked) {
                    for (const shape of track.shapes) {
                        if (!shapesByFrame.has(shape.frame)) {
                            const clonedShape = cloneDeep(shape);
                            targetTrack.shapes.push(clonedShape);
                            shapesByFrame.set(shape.frame, clonedShape);
                        }
                    }
                }
            }
        }

        const taskID = this.#instance instanceof Job ? this.#instance.taskId : this.#instance.id;
        if (taskID === null) {
            throw new Error('SAM2 tracker requires a task ID');
        }
        const job = this.#instance instanceof Job ? { job: this.#instance.id } : {};
        const jobKey = `${taskID}:${'job' in job ? job.job : 'task'}`;
        const sessionKey = `${jobKey}:${direction}`;
        const session = this.#sessions.get(sessionKey);
        const sourceObjects = sessionShapes.map(({ clientID }) => ({ clientID }));
        const objectKey = sessionShapes.length === 1 ? sessionShapes[0].clientID : 0;
        const sessionReused = session?.frame === number && isEqual(session.shapes, sessionShapes);
        let reusedContext = sessionReused;
        const previousWindow = [...this.#predictionWindows].reverse().find((window) => (
            window.jobKey === jobKey &&
            window.predictions.has(number) &&
            sourceObjects.every((object) => window.objects.some((candidate) => isEqual(candidate, object)))
        ));

        let states: TrackerResults['states'];
        let contextFrames = 1;
        let anchorFrame = number;
        if (sessionReused) {
            states = session.states;
            contextFrames = session.contextFrames;
            anchorFrame = session.anchorFrame;
        } else {
            onProgress('Initializing SAM2 tracker', 0);
            if (cancelled()) {
                return noChanges;
            }
            const startedAt = performance.now();
            const initialized = await core.lambda.call(taskID, this.#model, {
                type: 'init_tracking',
                frame: number,
                ...job,
                shapes: initialShapes,
            }) as SAM2TrackerResults;
            this.#logInference(
                'initialize', number, startedAt, initialShapes.length, objectKey, runID, direction, initialized, { frames: 0, anchor: number, reused: false },
            );

            if (!Array.isArray(initialized.states) || initialized.states.length !== initialShapes.length) {
                throw new Error('SAM2 tracker returned an invalid initialization response');
            }
            states = initialized.states;
        }
        if (cancelled()) {
            return noChanges;
        }

        const trackedShapes: Shape[] = [];
        let hasTrackedFrame = false;
        let endpointShapes: SessionShape[] | null = null;
        let trackedFrameCount = 0;
        let endpointFrame = number;
        const expectedTrackedFrames = Math.min(this.#frameCount, targetFrames.length);
        const predictions = new Map<number, SessionShape[]>();
        const supersededShapes: Shape[] = [];
        for (let index = 0; index < targetFrames.length; index++) {
            if (trackedFrameCount === this.#frameCount) {
                break;
            }
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

            const previousPredictions = previousWindow?.predictions.get(frame) ?? [];
            if (previousPredictions.length) {
                const statesAtFrame = await this.#instance.annotations.get(frame, false, []);
                const matches = statesAtFrame.filter((state) => (
                    state.objectType === ObjectType.SHAPE &&
                    state.source === Source.AUTO &&
                    previousPredictions.some((prediction) => (
                        state.shapeType === prediction.type &&
                        state.label.id === prediction.labelID &&
                        (state.group?.id ?? 0) === prediction.groupID &&
                        isEqual(state.points, prediction.points)
                    ))
                ));
                supersededShapes.push(...await Promise.all(matches.map((state) => state.export())) as Shape[]);
            }

            hasTrackedFrame = true;
            trackedFrameCount++;
            endpointFrame = frame;
            onProgress(
                `Tracking frame ${trackedFrameCount} of ${expectedTrackedFrames}`,
                Math.round(((trackedFrameCount - 1) / expectedTrackedFrames) * 100),
            );
            let result: SAM2TrackerResults;
            try {
                const startedAt = performance.now();
                result = await core.lambda.call(taskID, this.#model, {
                    type: 'track',
                    frame,
                    ...job,
                    states,
                }) as SAM2TrackerResults;
                this.#logInference(
                    'track', frame, startedAt, initialShapes.length, objectKey, runID, direction, result, { frames: contextFrames, anchor: anchorFrame, reused: reusedContext },
                );
            } catch (error) {
                this.#sessions.delete(sessionKey);
                throw error;
            }
            if (
                !Array.isArray(result.states) ||
                !Array.isArray(result.shapes) ||
                result.states.length !== initialShapes.length ||
                result.shapes.length !== initialShapes.length
            ) {
                throw new Error('SAM2 tracker returned an invalid tracking response');
            }
            states = result.states;
            contextFrames++;

            const correctedKeyframes = targetObjects.flatMap((targetObject, targetIndex) => {
                if (targetObjectStates[targetIndex].objectType !== ObjectType.TRACK) {
                    return [];
                }

                const existing = (targetObject as Track).shapes.find((shape) => shape.frame === frame);
                return existing && existing.source !== Source.AUTO && !existing.outside && existing.points?.length ? [{
                    targetIndex,
                    shape: {
                        type: targetObjectStates[targetIndex].shapeType,
                        points: [...existing.points],
                    },
                }] : [];
            });
            if (correctedKeyframes.length) {
                let reinitialized: SAM2TrackerResults;
                try {
                    const startedAt = performance.now();
                    reinitialized = await core.lambda.call(taskID, this.#model, {
                        type: 'init_tracking',
                        frame,
                        ...job,
                        shapes: correctedKeyframes.map(({ shape }) => shape),
                    }) as SAM2TrackerResults;
                    this.#logInference(
                        'correct', frame, startedAt, correctedKeyframes.length, objectKey, runID, direction, reinitialized, { frames: 0, anchor: frame, reused: false },
                    );
                } catch (error) {
                    this.#sessions.delete(sessionKey);
                    throw error;
                }
                if (
                    !Array.isArray(reinitialized.states) ||
                    reinitialized.states.length !== correctedKeyframes.length
                ) {
                    throw new Error('SAM2 tracker returned an invalid correction response');
                }

                for (let correctionIndex = 0; correctionIndex < correctedKeyframes.length; correctionIndex++) {
                    const { targetIndex, shape } = correctedKeyframes[correctionIndex];
                    states[targetIndex] = reinitialized.states[correctionIndex];
                    result.shapes[targetIndex] = shape;
                }
                contextFrames = 1;
                anchorFrame = frame;
                reusedContext = false;
            }

            endpointShapes = result.shapes.every((shape) => shape !== null) ?
                result.shapes.map((shape, shapeIndex) => ({
                    ...shape as MinimalShape,
                    points: [...(shape as MinimalShape).points],
                    clientID: sessionShapes[shapeIndex].clientID,
                    groupID: sessionShapes[shapeIndex].groupID,
                    labelID: sessionShapes[shapeIndex].labelID,
                })) : null;
            predictions.set(frame, result.shapes.flatMap((shape, shapeIndex) => (shape ? [{
                ...shape,
                points: [...shape.points],
                clientID: sessionShapes[shapeIndex].clientID,
                groupID: sessionShapes[shapeIndex].groupID,
                labelID: sessionShapes[shapeIndex].labelID,
            }] : [])));
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
                    if (existing && existing.source !== Source.AUTO) {
                        continue;
                    }
                    const updated = {
                        attributes: cloneDeep(existing?.attributes ?? []),
                        frame,
                        source: Source.AUTO,
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
                    track.frame = Math.min(track.frame, frame);
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

        if (direction === 1) {
            const nextFrame = frameNumbers
                .filter((frame) => frame > endpointFrame)
                .sort((left, right) => left - right)[0];
            if (Number.isInteger(nextFrame)) {
                for (const track of tracks) {
                    if (!track.shapes.some((shape) => shape.frame === nextFrame)) {
                        const lastShape = getTrackShape(track, endpointFrame);
                        if (lastShape) {
                            const { id: _serverID, ...position } = cloneDeep(lastShape);
                            track.shapes.push({
                                ...position,
                                frame: nextFrame,
                                outside: true,
                            });
                        }
                    }
                }
            }
        }

        onProgress('Tracking with SAM2', 100);
        if (endpointShapes) {
            this.#sessions.set(sessionKey, {
                frame: endpointFrame,
                shapes: endpointShapes,
                states,
                contextFrames,
                anchorFrame,
            });
        } else {
            this.#sessions.delete(sessionKey);
        }
        this.#predictionWindows.push({ jobKey, objects: sourceObjects, predictions });
        if (this.#predictionWindows.length > 20) {
            this.#predictionWindows.shift();
        }
        return {
            created: { shapes: trackedShapes, tags: [], tracks },
            deleted: {
                shapes: [
                    ...supersededShapes,
                    ...collection.shapes.filter((shape) => (
                        shape.type === ShapeType.MASK ||
                        (this.#convertPolygonShapesToTracks && shape.type === ShapeType.POLYGON)
                    )),
                ],
                tags: [],
                tracks: [...collection.tracks, ...linkedTracks],
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
                return shape && [ShapeType.MASK, ShapeType.POLYGON].includes(shape.type) && !shape.outside;
            }),
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        return [ShapeType.MASK, ShapeType.POLYGON].includes(objectState.shapeType);
    }

    public get name(): string {
        return SAM2_TRACKER_ACTION_NAME;
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
            'Frame count': {
                type: ActionParameterType.NUMBER,
                values: ['1', '100', '1'],
                defaultValue: '5',
            },
        };
    }
}
