// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';
import { RouteComponentProps } from 'react-router-dom';
import notification from 'antd/lib/notification';
import Progress from 'antd/lib/progress';
import { EventScope } from 'cvat-logger';

import {
    activateObject,
    changeFrameAsync,
    changeWorkspaceAsync,
    setHoveredChapter as setHoveredChapterAction,
    collectStatisticsAsync,
    deleteFrameAsync,
    redoActionAsync,
    restoreFrameAsync,
    saveAnnotationsAsync,
    searchAnnotationsAsync,
    searchChaptersAsync,
    setForceExitAnnotationFlag as setForceExitAnnotationFlagAction,
    setNavigationType as setNavigationTypeAction,
    showFilters as showFiltersAction,
    showStatistics as showStatisticsAction,
    switchNavigationBlocked as switchNavigationBlockedAction,
    switchPlay,
    switchShowSearchFramesModal as switchShowSearchFramesModalAction,
    undoActionAsync,
} from 'actions/annotation-actions';
import { registerComponentShortcuts } from 'actions/shortcuts-actions';
import AnnotationTopBarComponent from 'components/annotation-page/top-bar/top-bar';
import { Canvas } from 'cvat-canvas-wrapper';
import { Canvas3d } from 'cvat-canvas3d-wrapper';
import {
    BaseCollectionAction, FramesMetaData, getCore, Job, ObjectState, ShapeType,
} from 'cvat-core-wrapper';
import {
    ActiveControl, CombinedState, FrameSpeed, NavigationType, ToolsBlockerState, Workspace,
} from 'reducers';
import isAbleToChangeFrame from 'utils/is-able-to-change-frame';
import GlobalHotKeys, { KeyMap } from 'utils/mousetrap-react';
import { changeSAM2FrameCount, changeSAMTrackerModel, switchToolsBlockerState } from 'actions/settings-actions';
import { writeLatestFrame } from 'utils/remember-latest-frame';
import { finishDraw } from 'utils/drawing';
import { toClipboard } from 'utils/to-clipboard';
import { Chapter } from 'cvat-core/src/frames';
import {
    SAM2_TRACKER_ACTION_NAME, SAM2_TRACKER_MODEL_ID,
    SAM31_TRACKER_ACTION_NAME, SAM31_TRACKER_MODEL_ID,
} from 'utils/annotations-actions/sam2-tracker';
import { ShortcutScope } from 'utils/enums';
import { subKeyMap } from 'utils/component-subkeymap';

const core = getCore();
const componentShortcuts = {
    SAM2_TRACK_BACKWARD: {
        name: 'SAM: propagate frames backward',
        description: 'Track the selected polygon or mask backward',
        sequences: ['s'],
        scope: ShortcutScope.STANDARD_WORKSPACE,
    },
    SAM2_TRACK_FORWARD: {
        name: 'SAM: propagate frames forward',
        description: 'Track the selected polygon or mask forward',
        sequences: ['g'],
        scope: ShortcutScope.STANDARD_WORKSPACE,
    },
    SAM2_TRACK_ALL_BACKWARD: {
        name: 'SAM: propagate all objects backward',
        description: 'Track all visible, unlocked polygons and masks on the current frame backward',
        sequences: ['shift+s'],
        scope: ShortcutScope.STANDARD_WORKSPACE,
    },
    SAM2_TRACK_ALL_FORWARD: {
        name: 'SAM: propagate all objects forward',
        description: 'Track all visible, unlocked polygons and masks on the current frame forward',
        sequences: ['shift+g'],
        scope: ShortcutScope.STANDARD_WORKSPACE,
    },
};

registerComponentShortcuts(componentShortcuts);

interface StateToProps {
    chapters: Chapter[];
    hoveredChapter: number | null;
    jobInstance: Job;
    frameIsDeleted: boolean;
    frameNumber: number;
    frameFilename: string;
    frameStep: number;
    sam2FrameCount: number;
    samTrackerModelID: string;
    samTrackerModels: { value: string; label: string; disabled: boolean }[];
    frameSpeed: FrameSpeed;
    frameDelay: number;
    frameFetching: boolean;
    playing: boolean;
    saving: boolean;
    canvasIsReady: boolean;
    undoAction?: string;
    redoAction?: string;
    autoSave: boolean;
    autoSaveInterval: number;
    toolsBlockerState: ToolsBlockerState;
    showDeletedFrames: boolean;
    workspace: Workspace;
    keyMap: KeyMap;
    normalizedKeyMap: Record<string, string>;
    canvasInstance: Canvas | Canvas3d;
    forceExit: boolean;
    ranges: string;
    activeControl: ActiveControl;
    annotationFilters: object[];
    initialOpenGuide: boolean;
    navigationType: NavigationType;
    showSearchFrameByName: boolean;
    objectStates: ObjectState[];
    activatedStateID: number | null;
}

interface DispatchToProps {
    onChangeSAM2FrameCount(frameCount: number): void;
    onChangeSAMTrackerModel(modelID: string): void;
    onChangeFrame(frame: number, fillBuffer?: boolean, frameStep?: number): Promise<void>;
    activateSAM2Prediction(clientID: number): void;
    onSwitchPlay(playing: boolean): void;
    switchShowSearchPallet(visible: boolean): void;
    onSaveAnnotation(): void;
    showStatistics(sessionInstance: Job): void;
    showFilters(): void;
    undo(): void;
    redo(): void;
    searchAnnotations(
        sessionInstance: Job,
        frameFrom: number,
        frameTo: number,
        generalFilters?: {
            isEmptyFrame: boolean;
        },
    ): void;
    searchChapters(
        sessionInstance: Job,
        frameFrom: number,
        frameTo: number,
    ): void;
    setForceExitAnnotationFlag(forceExit: boolean): void;
    changeWorkspace(workspace: Workspace): void;
    setHoveredChapter(id: number | null): void;
    onSwitchToolsBlockerState(toolsBlockerState: ToolsBlockerState): void;
    deleteFrame(frame: number): void;
    restoreFrame(frame: number): void;
    switchNavigationBlocked(blocked: boolean): void;
    setNavigationType(navigationType: NavigationType): void;
}

function mapStateToProps(state: CombinedState): StateToProps {
    const {
        annotation: {
            player: {
                playing,
                ranges,
                frame: {
                    data: { deleted: frameIsDeleted },
                    filename: frameFilename,
                    number: frameNumber,
                    delay: frameDelay,
                    fetching: frameFetching,
                },
                navigationType,
                hoveredChapter,
            },
            annotations: {
                saving: { uploading: saving, forceExit },
                history,
                filters: annotationFilters,
                states: objectStates,
                activatedStateID,
            },
            job: { instance: jobInstance, queryParameters: { initialOpenGuide }, meta },
            canvas: { ready: canvasIsReady, instance: canvasInstance, activeControl },
            workspace,
        },
        settings: {
            player: {
                frameSpeed, frameStep, sam2FrameCount, samTrackerModelID, showDeletedFrames,
            },
            workspace: {
                autoSave,
                autoSaveInterval,
                toolsBlockerState,
            },
        },
        shortcuts: { keyMap, normalizedKeyMap },
    } = state;

    let showSearchFrameByName = false;
    if (meta?.frames && meta.frames.length > 0) {
        const firstName = meta.frames[0].name;
        showSearchFrameByName = !meta.frames.every(
            (frame: FramesMetaData['frames'][number]) => frame.name === firstName,
        );
    }

    const chapters = meta?.chapters ?? [];

    return {
        chapters,
        frameIsDeleted,
        frameStep,
        sam2FrameCount,
        samTrackerModelID: samTrackerModelID || SAM2_TRACKER_MODEL_ID,
        samTrackerModels: [
            { value: SAM2_TRACKER_MODEL_ID, label: 'SAM2 Tiny' },
            { value: SAM31_TRACKER_MODEL_ID, label: 'SAM3.1' },
        ].map((option) => ({
            ...option,
            disabled: !state.models.trackers.some((model) => model.id === option.value),
        })),
        frameSpeed,
        frameDelay,
        frameFetching,
        playing,
        canvasIsReady,
        hoveredChapter,
        saving,
        frameNumber,
        frameFilename,
        jobInstance: jobInstance as Job,
        undoAction: history.undo.length ? history.undo[history.undo.length - 1][0] : undefined,
        redoAction: history.redo.length ? history.redo[history.redo.length - 1][0] : undefined,
        autoSave,
        autoSaveInterval,
        toolsBlockerState,
        showDeletedFrames,
        workspace,
        keyMap,
        normalizedKeyMap,
        canvasInstance: canvasInstance as NonNullable<typeof canvasInstance>,
        forceExit,
        activeControl,
        ranges,
        annotationFilters,
        initialOpenGuide,
        navigationType,
        showSearchFrameByName,
        objectStates,
        activatedStateID,
    };
}

function mapDispatchToProps(dispatch: any): DispatchToProps {
    return {
        onChangeSAM2FrameCount(frameCount: number): void {
            dispatch(changeSAM2FrameCount(frameCount));
        },
        onChangeSAMTrackerModel(modelID: string): void {
            dispatch(changeSAMTrackerModel(modelID));
        },
        onChangeFrame(frame: number, fillBuffer?: boolean, frameStep?: number): Promise<void> {
            return dispatch(changeFrameAsync(frame, fillBuffer, frameStep));
        },
        activateSAM2Prediction(clientID: number): void {
            dispatch((innerDispatch: any, getState: () => CombinedState) => {
                const prediction = getState().annotation.annotations.states
                    .find((state) => state.clientID === clientID);
                if (prediction) {
                    innerDispatch(activateObject(prediction.clientID, null, null));
                }
            });
        },
        onSwitchPlay(playing: boolean): void {
            dispatch(switchPlay(playing));
        },
        onSaveAnnotation(): void {
            dispatch(saveAnnotationsAsync());
        },
        showStatistics(sessionInstance: Job): void {
            dispatch(collectStatisticsAsync(sessionInstance));
            dispatch(showStatisticsAction(true));
        },
        showFilters(): void {
            dispatch(showFiltersAction(true));
        },
        switchShowSearchPallet(visible: boolean): void {
            dispatch(switchShowSearchFramesModalAction(visible));
        },
        undo(): void {
            dispatch(undoActionAsync());
        },
        redo(): void {
            dispatch(redoActionAsync());
        },
        searchAnnotations(
            sessionInstance: Job,
            frameFrom: number,
            frameTo: number,
            generalFilters?: {
                isEmptyFrame: boolean;
            },
        ): void {
            dispatch(searchAnnotationsAsync(sessionInstance, frameFrom, frameTo, generalFilters));
        },
        searchChapters(
            sessionInstance: Job,
            frameFrom: number,
            frameTo: number,
        ) {
            dispatch(searchChaptersAsync(sessionInstance, frameFrom, frameTo));
        },
        changeWorkspace(workspace: Workspace): void {
            dispatch(changeWorkspaceAsync(workspace));
        },
        setHoveredChapter(id: number | null) {
            dispatch(setHoveredChapterAction(id));
        },
        setForceExitAnnotationFlag(forceExit: boolean): void {
            dispatch(setForceExitAnnotationFlagAction(forceExit));
        },
        onSwitchToolsBlockerState(toolsBlockerState: ToolsBlockerState): void {
            dispatch(switchToolsBlockerState(toolsBlockerState));
        },
        deleteFrame(frame: number): void {
            dispatch(deleteFrameAsync(frame));
        },
        restoreFrame(frame: number): void {
            dispatch(restoreFrameAsync(frame));
        },
        switchNavigationBlocked(blocked: boolean): void {
            dispatch(switchNavigationBlockedAction(blocked));
        },
        setNavigationType(navigationType: NavigationType): void {
            dispatch(setNavigationTypeAction(navigationType));
        },
    };
}

type Props = StateToProps & DispatchToProps & RouteComponentProps;
class AnnotationTopBarContainer extends React.PureComponent<Props> {
    private inputFrameRef: React.RefObject<HTMLInputElement>;
    private autoSaveInterval: number | undefined;
    private isWaitingForPlayDelay: boolean;
    private unblock: any;

    constructor(props: Props) {
        super(props);
        this.isWaitingForPlayDelay = false;
        this.inputFrameRef = React.createRef<HTMLInputElement>();
    }

    public componentDidMount(): void {
        const {
            autoSaveInterval, history, jobInstance, setForceExitAnnotationFlag,
        } = this.props;
        this.autoSaveInterval = window.setInterval(this.autoSave.bind(this), autoSaveInterval);

        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const self = this;
        this.unblock = history.block((location: any) => {
            const { forceExit, frameNumber } = self.props;
            const { id: jobID, taskId: taskID } = jobInstance;
            writeLatestFrame(jobInstance.id, frameNumber);

            if (
                jobInstance.annotations.hasUnsavedChanges() &&
                location.pathname !== `/tasks/${taskID}/jobs/${jobID}` &&
                !forceExit
            ) {
                return 'You have unsaved changes, please confirm leaving this page.';
            }

            if (forceExit) {
                setForceExitAnnotationFlag(false);
            }

            return undefined;
        });

        window.addEventListener('beforeunload', this.beforeUnloadCallback);
    }

    public componentDidUpdate(prevProps: Props): void {
        const { autoSaveInterval } = this.props;

        if (autoSaveInterval !== prevProps.autoSaveInterval) {
            if (this.autoSaveInterval) window.clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = window.setInterval(this.autoSave.bind(this), autoSaveInterval);
        }
        this.handlePlayIfNecessary();
    }

    public componentWillUnmount(): void {
        window.clearInterval(this.autoSaveInterval);
        window.removeEventListener('beforeunload', this.beforeUnloadCallback);
        this.unblock();
    }

    private async handlePlayIfNecessary(): Promise<void> {
        const {
            jobInstance,
            frameNumber,
            frameDelay,
            frameFetching,
            playing,
            canvasIsReady,
            onSwitchPlay,
            onChangeFrame,
        } = this.props;

        const { stopFrame } = jobInstance;

        if (playing && canvasIsReady && !frameFetching && !this.isWaitingForPlayDelay) {
            this.isWaitingForPlayDelay = true;
            try {
                await new Promise((resolve) => {
                    setTimeout(resolve, frameDelay);
                });

                const { playing: currentPlaying, showDeletedFrames } = this.props;

                if (currentPlaying) {
                    const nextCandidate = frameNumber + 1;
                    if (nextCandidate > stopFrame) {
                        onSwitchPlay(false);
                        return;
                    }

                    const next = await jobInstance.frames
                        .search({ notDeleted: !showDeletedFrames }, nextCandidate, stopFrame);
                    if (next !== null && isAbleToChangeFrame(next)) {
                        onChangeFrame(next, currentPlaying);
                    } else {
                        onSwitchPlay(false);
                    }
                }
            } finally {
                this.isWaitingForPlayDelay = false;
            }
        }
    }

    private undo = (): void => {
        const { undo, undoAction } = this.props;

        if (isAbleToChangeFrame() && undoAction) {
            undo();
        }
    };

    private redo = (): void => {
        const { redo, redoAction } = this.props;

        if (isAbleToChangeFrame() && redoAction) {
            redo();
        }
    };

    private showStatistics = (): void => {
        const { jobInstance, showStatistics } = this.props;
        showStatistics(jobInstance);
    };

    private showFilters = (): void => {
        const { showFilters } = this.props;
        showFilters();
    };

    private onSwitchPlay = (): void => {
        const {
            frameNumber, jobInstance, onSwitchPlay, playing,
        } = this.props;

        if (playing) {
            onSwitchPlay(false);
        } else if (frameNumber < jobInstance.stopFrame) {
            onSwitchPlay(true);
        }
    };

    private onFirstFrame = async (): Promise<void> => {
        const {
            frameNumber, jobInstance, playing,
            onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame =
            await jobInstance.frames.search({ notDeleted: !showDeletedFrames }, jobInstance.startFrame, frameNumber);
        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onBackward = async (): Promise<void> => {
        const {
            frameNumber, frameStep, jobInstance, playing,
            onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames, offset: frameStep },
            Math.max(jobInstance.startFrame, frameNumber - 1),
            jobInstance.startFrame,
        );

        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onPrevFrame = async (): Promise<void> => {
        const {
            frameNumber, jobInstance, playing, searchAnnotations,
            onSwitchPlay, showDeletedFrames, navigationType, searchChapters,
        } = this.props;
        const { startFrame } = jobInstance;

        const frameFrom = Math.max(jobInstance.startFrame, frameNumber - 1);
        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames },
            frameFrom,
            jobInstance.startFrame,
        );

        if (newFrame !== frameNumber && newFrame !== null && isAbleToChangeFrame(newFrame)) {
            if (playing) {
                onSwitchPlay(false);
            }

            if (navigationType === NavigationType.REGULAR) {
                this.changeFrame(newFrame);
            } else if (navigationType === NavigationType.FILTERED) {
                searchAnnotations(jobInstance, newFrame, startFrame);
            } else if (navigationType === NavigationType.CHAPTER) {
                searchChapters(jobInstance, newFrame, startFrame);
            } else {
                searchAnnotations(jobInstance, newFrame, startFrame, { isEmptyFrame: true });
            }
        }
    };

    private onNextFrame = async (): Promise<void> => {
        const {
            frameNumber, jobInstance, playing, searchAnnotations, searchChapters,
            onSwitchPlay, showDeletedFrames, navigationType,
        } = this.props;
        const { stopFrame } = jobInstance;

        const frameFrom = Math.min(jobInstance.stopFrame, frameNumber + 1);
        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames },
            frameFrom,
            jobInstance.stopFrame,
        );
        if (newFrame !== frameNumber && newFrame !== null && isAbleToChangeFrame(newFrame)) {
            if (playing) {
                onSwitchPlay(false);
            }

            if (navigationType === NavigationType.REGULAR) {
                this.changeFrame(newFrame);
            } else if (navigationType === NavigationType.FILTERED) {
                searchAnnotations(jobInstance, newFrame, stopFrame);
            } else if (navigationType === NavigationType.CHAPTER) {
                searchChapters(jobInstance, newFrame, stopFrame);
            } else {
                searchAnnotations(jobInstance, newFrame, stopFrame, { isEmptyFrame: true });
            }
        }
    };

    private trackingSAM2 = false;

    private onTrackSAM2 = async (direction: -1 | 1, allObjects = false): Promise<void> => {
        const {
            activateSAM2Prediction, activatedStateID, frameNumber, jobInstance, objectStates,
            playing, onSwitchPlay, sam2FrameCount, frameFetching, frameIsDeleted,
            samTrackerModelID = SAM2_TRACKER_MODEL_ID,
        } = this.props;
        if (this.trackingSAM2 || frameFetching || frameIsDeleted || !isAbleToChangeFrame()) {
            return;
        }

        const eligibleStates = objectStates
            .filter((state) => (
                state.frame === frameNumber && !state.outside && !state.lock && !state.hidden &&
                [ShapeType.POLYGON, ShapeType.MASK].includes(state.shapeType)
            ))
            .sort((left, right) => right.updated - left.updated);
        const selectedState = objectStates.find((state) => state.clientID === activatedStateID) ?? eligibleStates[0];
        const trackingStates = allObjects ? eligibleStates : eligibleStates.filter((state) => state === selectedState);
        if (!trackingStates.length) {
            notification.warning({ message: 'Draw or select a visible, unlocked polygon or mask before tracking' });
            return;
        }
        const primaryObjectIndex = Math.max(trackingStates.indexOf(selectedState), 0);
        const objectState = trackingStates[primaryObjectIndex];
        const objectCount = trackingStates.length;
        const objectDescription = `${objectCount} object${objectCount === 1 ? '' : 's'}`;
        const modelName = samTrackerModelID === SAM31_TRACKER_MODEL_ID ? 'SAM3.1' : 'SAM2';
        const actionName = samTrackerModelID === SAM31_TRACKER_MODEL_ID ?
            SAM31_TRACKER_ACTION_NAME : SAM2_TRACKER_ACTION_NAME;

        this.trackingSAM2 = true;
        const trackingStartedAt = performance.now();
        let trackingAttempted = false;
        let cancelled = false;
        let progressFinished = false;
        const directionLabel = direction === 1 ? 'forward' : 'backward';
        const progressKey = `sam2-tracking-${jobInstance.id}-${directionLabel}`;
        const cancellationKey = `${progressKey}-cancel`;
        const showProgress = (message: string, percent: number): void => {
            if (cancelled || progressFinished) {
                return;
            }
            if (percent >= 100) {
                // Ant Design also invokes onClose when we destroy a completed notification.
                progressFinished = true;
                notification.destroy(progressKey);
                return;
            }

            notification.info({
                key: progressKey,
                message: `${modelName} tracking ${objectDescription} ${directionLabel}`,
                description: (
                    <>
                        <Progress percent={percent} size='small' status='active' />
                        {message}
                        <div>Close this notification to cancel tracking.</div>
                    </>
                ),
                duration: 0,
                placement: 'bottomRight',
                onClose: () => {
                    if (!progressFinished && !cancelled) {
                        cancelled = true;
                        notification.info({
                            key: cancellationKey,
                            message: `Cancelling ${modelName} tracking…`,
                            description: 'Waiting for the current request to finish. No more frames will be requested.',
                            duration: 0,
                            closable: false,
                            placement: 'bottomRight',
                        });
                    }
                },
            });
        };
        showProgress(`Preparing up to ${sam2FrameCount} frames`, 0);
        try {
            const action = (await core.actions.list()).find((item) => item.name === actionName);
            if (cancelled) {
                return;
            }
            if (!(action instanceof BaseCollectionAction) ||
                !trackingStates.every((state) => action.isApplicableForObject(state))) {
                notification.error({ key: progressKey, message: `${modelName} tracker is unavailable` });
                return;
            }

            const targetFrame = direction === 1 ? jobInstance.stopFrame : jobInstance.startFrame;
            const firstPredictedFrame = frameNumber === targetFrame ? null : await jobInstance.frames.search(
                { notDeleted: true },
                frameNumber + direction,
                targetFrame,
            );
            if (cancelled) {
                return;
            }
            if (
                firstPredictedFrame === null ||
                firstPredictedFrame === frameNumber ||
                !isAbleToChangeFrame(firstPredictedFrame)
            ) {
                notification.warning({
                    key: progressKey,
                    message: `There is no frame to track ${direction === 1 ? 'forward' : 'backward'}`,
                });
                return;
            }
            trackingAttempted = true;
            await core.actions.call(jobInstance, action, {
                'Convert polygon shapes to tracks': 'false',
                'Target frame': String(targetFrame),
                'Frame count': String(sam2FrameCount),
            }, frameNumber, trackingStates, showProgress, () => cancelled);
            if (cancelled) {
                return;
            }
            showProgress(`Tracking with ${modelName}`, 100);
            jobInstance.logger.log(EventScope.sam2Tracking, {
                duration: Math.round(performance.now() - trackingStartedAt),
                outcome: 'success',
                direction: directionLabel,
                requested_frames: sam2FrameCount,
                object_count: objectCount,
                model_id: samTrackerModelID,
                video_name: jobInstance.taskName || `Task ${jobInstance.taskId}`,
            });
            if (playing) {
                onSwitchPlay(false);
            }
            await this.changeFrame(firstPredictedFrame);
            activateSAM2Prediction(objectState.clientID);
            notification.success({
                key: progressKey,
                message: `${modelName} predictions ready for ${objectDescription} ${directionLabel}`,
                description: `Opened frame ${firstPredictedFrame}`,
                duration: 2,
                placement: 'bottomRight',
            });
        } catch (error) {
            if (cancelled) {
                return;
            }
            if (trackingAttempted) {
                jobInstance.logger.log(EventScope.sam2Tracking, {
                    duration: Math.round(performance.now() - trackingStartedAt),
                    outcome: 'failed',
                    direction: directionLabel,
                    requested_frames: sam2FrameCount,
                    object_count: objectCount,
                    model_id: samTrackerModelID,
                    video_name: jobInstance.taskName || `Task ${jobInstance.taskId}`,
                    error_type: error instanceof Error ? error.name : 'unknown',
                });
            }
            notification.error({
                key: progressKey,
                message: error instanceof Error ? error.message : String(error),
            });
        } finally {
            progressFinished = true;
            this.trackingSAM2 = false;
            if (cancelled) {
                notification.info({
                    key: cancellationKey,
                    message: `${modelName} tracking cancelled`,
                    description: 'Predictions from this run were discarded. You can start tracking again.',
                    duration: 3,
                    placement: 'bottomRight',
                });
            }
        }
    };

    private onForward = async (): Promise<void> => {
        const {
            frameNumber, frameStep, jobInstance, playing,
            onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames, offset: frameStep },
            Math.min(jobInstance.stopFrame, frameNumber + 1),
            jobInstance.stopFrame,
        );

        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private onLastFrame = async (): Promise<void> => {
        const {
            frameNumber, jobInstance, playing,
            onSwitchPlay, showDeletedFrames,
        } = this.props;

        const newFrame =
            await jobInstance.frames.search({ notDeleted: !showDeletedFrames }, jobInstance.stopFrame, frameNumber);
        if (newFrame !== frameNumber && newFrame !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(newFrame);
        }
    };

    private searchAnnotations = (direction: 'forward' | 'backward'): void => {
        const {
            frameNumber, jobInstance, searchAnnotations,
        } = this.props;
        const { startFrame, stopFrame } = jobInstance;

        if (isAbleToChangeFrame()) {
            if (direction === 'forward' && frameNumber + 1 <= stopFrame) {
                searchAnnotations(jobInstance, frameNumber + 1, stopFrame);
            } else if (direction === 'backward' && frameNumber - 1 >= startFrame) {
                searchAnnotations(jobInstance, frameNumber - 1, startFrame);
            }
        }
    };

    private readonly searchChapters = (direction: 'forward' | 'backward'): void => {
        const {
            frameNumber, jobInstance, searchChapters,
        } = this.props;
        const { startFrame, stopFrame } = jobInstance;

        if (isAbleToChangeFrame()) {
            if (direction === 'forward' && frameNumber + 1 <= stopFrame) {
                searchChapters(jobInstance, frameNumber + 1, stopFrame);
            } else if (direction === 'backward' && frameNumber - 1 >= startFrame) {
                searchChapters(jobInstance, frameNumber - 1, startFrame);
            }
        }
    };

    private readonly selectChapter = async (id: number): Promise<void> => {
        const {
            chapters, playing, onSwitchPlay,
        } = this.props;

        const selectedChapter = chapters.find((chapter: Chapter) => chapter.id === id) ?? null;

        if (selectedChapter !== null) {
            if (playing) {
                onSwitchPlay(false);
            }
            this.changeFrame(selectedChapter.start);
        }
    };

    private onChangePlayerSliderValue = async (value: number): Promise<void> => {
        const {
            playing, onSwitchPlay, jobInstance, showDeletedFrames,
        } = this.props;
        if (playing) {
            onSwitchPlay(false);
        }
        const newFrame = await jobInstance.frames.search(
            { notDeleted: !showDeletedFrames },
            Math.min(jobInstance.stopFrame, value),
            jobInstance.stopFrame,
        );
        if (newFrame !== null) {
            this.changeFrame(newFrame);
        }
    };

    private onChangePlayerInputValue = async (value: number): Promise<void> => {
        const {
            frameNumber, onSwitchPlay, playing, jobInstance, showDeletedFrames,
        } = this.props;

        if (value !== frameNumber) {
            if (playing) {
                onSwitchPlay(false);
            }
            const newFrame = await jobInstance.frames.search(
                { notDeleted: !showDeletedFrames },
                Math.min(jobInstance.stopFrame, value),
                jobInstance.stopFrame,
            );
            if (newFrame !== null) {
                this.changeFrame(newFrame);
            }
        }
    };

    private onFinishDraw = (): void => {
        const { activeControl, canvasInstance } = this.props;
        finishDraw(canvasInstance, activeControl);
    };

    private onSwitchToolsBlockerState = (): void => {
        const { toolsBlockerState, onSwitchToolsBlockerState } = this.props;
        onSwitchToolsBlockerState({ algorithmsLocked: !toolsBlockerState.algorithmsLocked });
    };

    private onURLIconClick = (): void => {
        const { frameNumber } = this.props;
        const { origin, pathname } = window.location;
        const url = `${origin}${pathname}?frame=${frameNumber}`;

        toClipboard(url);
    };

    private onCopyFilenameIconClick = (): void => {
        const { frameFilename } = this.props;

        toClipboard(frameFilename);
    };

    private onDeleteFrame = (): void => {
        const { deleteFrame, frameNumber } = this.props;
        deleteFrame(frameNumber);
    };

    private onRestoreFrame = (): void => {
        const { restoreFrame, frameNumber } = this.props;
        restoreFrame(frameNumber);
    };

    private changeWorkspace = (workspace: Workspace): void => {
        const { changeWorkspace } = this.props;
        changeWorkspace(workspace);
        if (window.document.activeElement) {
            (window.document.activeElement as HTMLElement).blur();
        }
    };

    private readonly setHoveredChapter = (id: number | null): void => {
        const { setHoveredChapter } = this.props;
        setHoveredChapter(id);
    };

    private beforeUnloadCallback = (event: BeforeUnloadEvent): string | undefined => {
        const { jobInstance, forceExit, setForceExitAnnotationFlag } = this.props;
        const { frameNumber } = this.props;

        writeLatestFrame(jobInstance.id, frameNumber);
        if (jobInstance.annotations.hasUnsavedChanges() && !forceExit) {
            const confirmationMessage = 'You have unsaved changes, please confirm leaving this page.';

            // eslint-disable-next-line no-param-reassign
            event.returnValue = confirmationMessage;
            return confirmationMessage;
        }

        if (forceExit) {
            setForceExitAnnotationFlag(false);
        }
        return undefined;
    };

    private autoSave(): void {
        const { autoSave, saving, onSaveAnnotation } = this.props;

        if (autoSave && !saving) {
            onSaveAnnotation();
        }
    }

    private changeFrame(frame: number): Promise<void> {
        const { onChangeFrame } = this.props;
        if (isAbleToChangeFrame(frame)) {
            return onChangeFrame(frame);
        }
        return Promise.resolve();
    }

    public render(): JSX.Element {
        const {
            playing,
            saving,
            chapters,
            hoveredChapter,
            jobInstance,
            jobInstance: { startFrame, stopFrame },
            frameNumber,
            frameFilename,
            frameIsDeleted,
            undoAction,
            redoAction,
            workspace,
            keyMap,
            ranges,
            normalizedKeyMap,
            activeControl,
            annotationFilters,
            initialOpenGuide,
            toolsBlockerState,
            navigationType,
            switchNavigationBlocked,
            setNavigationType,
            switchShowSearchPallet,
            showSearchFrameByName,
            sam2FrameCount,
            onChangeSAM2FrameCount,
            samTrackerModelID,
            samTrackerModels,
            onChangeSAMTrackerModel,
        } = this.props;
        const sam2KeyMap = workspace === Workspace.STANDARD ? subKeyMap(componentShortcuts, keyMap) : {};

        const topBar = (
            <AnnotationTopBarComponent
                showStatistics={this.showStatistics}
                sam2FrameCount={sam2FrameCount}
                onChangeSAM2FrameCount={onChangeSAM2FrameCount}
                samTrackerModelID={samTrackerModelID}
                samTrackerModels={samTrackerModels}
                onChangeSAMTrackerModel={onChangeSAMTrackerModel}
                showFilters={this.showFilters}
                onSwitchPlay={this.onSwitchPlay}
                onPrevFrame={this.onPrevFrame}
                onNextFrame={this.onNextFrame}
                onForward={this.onForward}
                onBackward={this.onBackward}
                onFirstFrame={this.onFirstFrame}
                onLastFrame={this.onLastFrame}
                onSearchAnnotations={this.searchAnnotations}
                onSearchChapters={this.searchChapters}
                onSelectChapter={this.selectChapter}
                setHoveredChapter={this.setHoveredChapter}
                setNavigationType={setNavigationType}
                onSliderChange={this.onChangePlayerSliderValue}
                onInputChange={this.onChangePlayerInputValue}
                onURLIconClick={this.onURLIconClick}
                onCopyFilenameIconClick={this.onCopyFilenameIconClick}
                onDeleteFrame={this.onDeleteFrame}
                onRestoreFrame={this.onRestoreFrame}
                changeWorkspace={this.changeWorkspace}
                switchShowSearchPallet={switchShowSearchPallet}
                showSearchFrameByName={showSearchFrameByName}
                switchNavigationBlocked={switchNavigationBlocked}
                keyMap={keyMap}
                workspace={workspace}
                playing={playing}
                chapters={chapters}
                hoveredChapter={hoveredChapter}
                saving={saving}
                ranges={ranges}
                startFrame={startFrame}
                stopFrame={stopFrame}
                frameNumber={frameNumber}
                frameFilename={frameFilename}
                frameDeleted={frameIsDeleted}
                inputFrameRef={this.inputFrameRef}
                undoAction={undoAction}
                redoAction={redoAction}
                undoShortcut={normalizedKeyMap.UNDO}
                redoShortcut={normalizedKeyMap.REDO}
                drawShortcut={normalizedKeyMap.SWITCH_DRAW_MODE_STANDARD_CONTROLS}
                switchToolsBlockerShortcut={normalizedKeyMap.SWITCH_TOOLS_BLOCKER_STATE}
                playPauseShortcut={normalizedKeyMap.PLAY_PAUSE}
                deleteFrameShortcut={normalizedKeyMap.DELETE_FRAME}
                nextFrameShortcut={normalizedKeyMap.NEXT_FRAME}
                previousFrameShortcut={normalizedKeyMap.PREV_FRAME}
                forwardShortcut={normalizedKeyMap.FORWARD_FRAME}
                backwardShortcut={normalizedKeyMap.BACKWARD_FRAME}
                navigationType={navigationType}
                focusFrameInputShortcut={normalizedKeyMap.FOCUS_INPUT_FRAME}
                searchFrameByNameShortcut={normalizedKeyMap.SEARCH_FRAME_BY_NAME}
                annotationFilters={annotationFilters}
                initialOpenGuide={initialOpenGuide}
                onUndoClick={this.undo}
                onRedoClick={this.redo}
                onFinishDraw={this.onFinishDraw}
                onSwitchToolsBlockerState={this.onSwitchToolsBlockerState}
                toolsBlockerState={toolsBlockerState}
                jobInstance={jobInstance}
                activeControl={activeControl}
            />
        );

        return (
            <>
                <GlobalHotKeys
                    keyMap={sam2KeyMap}
                    handlers={{
                        SAM2_TRACK_BACKWARD: () => { this.onTrackSAM2(-1); },
                        SAM2_TRACK_FORWARD: () => { this.onTrackSAM2(1); },
                        SAM2_TRACK_ALL_BACKWARD: () => { this.onTrackSAM2(-1, true); },
                        SAM2_TRACK_ALL_FORWARD: () => { this.onTrackSAM2(1, true); },
                    }}
                />
                {topBar}
            </>
        );
    }
}

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(AnnotationTopBarContainer));
