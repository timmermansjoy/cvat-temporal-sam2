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
    BaseCollectionAction, FramesMetaData, getCore, Job, JobType, ObjectState, ShapeType, Source, Task,
} from 'cvat-core-wrapper';
import {
    ActiveControl, CombinedState, FrameSpeed, NavigationType, ToolsBlockerState, Workspace,
} from 'reducers';
import isAbleToChangeFrame from 'utils/is-able-to-change-frame';
import GlobalHotKeys, { KeyMap } from 'utils/mousetrap-react';
import { switchToolsBlockerState } from 'actions/settings-actions';
import { writeLatestFrame } from 'utils/remember-latest-frame';
import { finishDraw } from 'utils/drawing';
import { toClipboard } from 'utils/to-clipboard';
import { Chapter } from 'cvat-core/src/frames';
import {
    SAM2_TRACKER_ACTION_NAME, SAM2_TRACKER_MODEL_ID,
} from 'utils/annotations-actions/sam2-tracker';
import { ShortcutScope } from 'utils/enums';
import { subKeyMap } from 'utils/component-subkeymap';

const core = getCore();
const componentShortcuts = {
    SAM2_TRACK_BACKWARD: {
        name: 'SAM2: propagate frames backward',
        description: 'Track the selected polygon or mask backward',
        sequences: ['s'],
        scope: ShortcutScope.STANDARD_WORKSPACE,
    },
    SAM2_TRACK_FORWARD: {
        name: 'SAM2: propagate frames forward',
        description: 'Track the selected polygon or mask forward',
        sequences: ['g'],
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
                frameSpeed, frameStep, sam2FrameCount, showDeletedFrames,
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
type SAM2NavigationState = {
    sam2Prediction?: {
        groupID: number | null;
        labelID: number;
    };
};

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
        this.activatePendingSAM2Prediction();
    }

    public componentDidUpdate(prevProps: Props): void {
        const { autoSaveInterval } = this.props;

        if (autoSaveInterval !== prevProps.autoSaveInterval) {
            if (this.autoSaveInterval) window.clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = window.setInterval(this.autoSave.bind(this), autoSaveInterval);
        }
        this.activatePendingSAM2Prediction();
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

    private activatePendingSAM2Prediction = (): void => {
        const {
            activateSAM2Prediction, frameNumber, history, location, objectStates,
        } = this.props;
        const pending = (location.state as SAM2NavigationState | undefined)?.sam2Prediction;
        if (!pending) {
            return;
        }

        const prediction = objectStates
            .filter((state) => (
                state.frame === frameNumber &&
                !state.outside &&
                state.label.id === pending.labelID &&
                [ShapeType.POLYGON, ShapeType.MASK].includes(state.shapeType) &&
                (
                    pending.groupID ?
                        state.group?.id === pending.groupID :
                        state.source === Source.AUTO
                )
            ))
            .sort((left, right) => right.updated - left.updated)[0];
        if (typeof prediction?.clientID === 'number') {
            activateSAM2Prediction(prediction.clientID);
            history.replace(`${location.pathname}${location.search}${location.hash}`);
        }
    };

    private onTrackSAM2 = async (direction: -1 | 1): Promise<void> => {
        const {
            activateSAM2Prediction, activatedStateID, frameNumber, jobInstance, objectStates,
            history, playing, onSwitchPlay, sam2FrameCount,
        } = this.props;
        if (this.trackingSAM2) {
            return;
        }

        const objectState = objectStates.find((state) => state.clientID === activatedStateID) ?? objectStates
            .filter((state) => (
                state.frame === frameNumber && [ShapeType.POLYGON, ShapeType.MASK].includes(state.shapeType)
            ))
            .sort((left, right) => right.updated - left.updated)[0];
        if (
            !objectState ||
            objectState.outside ||
            ![ShapeType.POLYGON, ShapeType.MASK].includes(objectState.shapeType)
        ) {
            notification.warning({ message: 'Draw or select a polygon or mask before tracking' });
            return;
        }

        this.trackingSAM2 = true;
        const trackingStartedAt = performance.now();
        let trackingAttempted = false;
        const directionLabel = direction === 1 ? 'forward' : 'backward';
        const progressKey = `sam2-tracking-${jobInstance.id}-${directionLabel}`;
        const showProgress = (message: string, percent: number): void => {
            if (percent >= 100) {
                notification.destroy(progressKey);
                return;
            }

            notification.info({
                key: progressKey,
                message: `SAM2 tracking ${directionLabel}`,
                description: (
                    <>
                        <Progress percent={percent} size='small' status='active' />
                        {message}
                    </>
                ),
                duration: 0,
                placement: 'bottomRight',
            });
        };
        showProgress(`Preparing up to ${sam2FrameCount} frames`, 0);
        try {
            const action = (await core.actions.list()).find((item) => item.name === SAM2_TRACKER_ACTION_NAME);
            if (!(action instanceof BaseCollectionAction) || !action.isApplicableForObject(objectState)) {
                notification.error({ key: progressKey, message: 'SAM2 tracker is unavailable' });
                return;
            }

            const currentJobFrames = (await jobInstance.frames.frameNumbers())
                .filter((frame) => (direction === 1 ? frame > frameNumber : frame < frameNumber))
                .sort((left, right) => direction * (left - right));
            let availableCurrentJobFrames = 0;
            for (const frame of currentJobFrames) {
                if (!(await jobInstance.frames.get(frame)).deleted) {
                    availableCurrentJobFrames++;
                    if (availableCurrentJobFrames === sam2FrameCount) {
                        break;
                    }
                }
            }

            let actionInstance: Job | Task = jobInstance;
            let actionObjectState = objectState;
            let targetFrame = direction === 1 ? jobInstance.stopFrame : jobInstance.startFrame;
            let destinationJob = jobInstance;
            if (availableCurrentJobFrames < sam2FrameCount && jobInstance.taskId !== null) {
                const [taskInstance] = await core.tasks.get({ id: jobInstance.taskId });
                const taskJobs = taskInstance.jobs
                    .filter((job) => job.type === JobType.ANNOTATION && job.parentJobId === null)
                    .sort((left, right) => left.startFrame - right.startFrame);
                const taskBoundary = direction === 1 ?
                    Math.max(...taskJobs.map((job) => job.stopFrame)) :
                    Math.min(...taskJobs.map((job) => job.startFrame));
                if (
                    (direction === 1 && taskBoundary > jobInstance.stopFrame) ||
                    (direction === -1 && taskBoundary < jobInstance.startFrame)
                ) {
                    await jobInstance.annotations.save();
                    const refreshedJobState = (await jobInstance.annotations.get(frameNumber, false, []))
                        .find((state) => state.clientID === objectState.clientID);
                    const taskStates = await taskInstance.annotations.get(frameNumber, false, []);
                    const matchingTaskState = taskStates.find((state) => (
                        refreshedJobState?.serverID !== null &&
                        state.serverID === refreshedJobState?.serverID
                    ));
                    if (!matchingTaskState) {
                        throw new Error('Could not continue the selected object across jobs');
                    }

                    actionInstance = taskInstance;
                    actionObjectState = matchingTaskState;
                    targetFrame = taskBoundary;
                }
            }

            const firstPredictedFrame = await actionInstance.frames.search(
                { notDeleted: true },
                frameNumber + direction,
                targetFrame,
            );
            if (
                firstPredictedFrame === null ||
                firstPredictedFrame === frameNumber
            ) {
                notification.warning({
                    key: progressKey,
                    message: `There is no frame to track ${direction === 1 ? 'forward' : 'backward'}`,
                });
                return;
            }
            if (actionInstance instanceof Task) {
                const matchingJob = actionInstance.jobs.find((job) => (
                    job.type === JobType.ANNOTATION &&
                    job.parentJobId === null &&
                    firstPredictedFrame >= job.startFrame &&
                    firstPredictedFrame <= job.stopFrame
                ));
                if (!matchingJob) {
                    throw new Error('The next frame is not available in an annotation job');
                }
                destinationJob = matchingJob;
            } else if (!isAbleToChangeFrame(firstPredictedFrame)) {
                notification.warning({
                    key: progressKey,
                    message: `There is no frame to track ${direction === 1 ? 'forward' : 'backward'}`,
                });
                return;
            }

            trackingAttempted = true;
            await core.actions.call(actionInstance, action, {
                'Convert polygon shapes to tracks': 'false',
                'Target frame': String(targetFrame),
                'Frame count': String(sam2FrameCount),
            }, frameNumber, [actionObjectState], showProgress, () => false);
            jobInstance.logger.log(EventScope.sam2Tracking, {
                duration: Math.round(performance.now() - trackingStartedAt),
                outcome: 'success',
                direction: directionLabel,
                requested_frames: sam2FrameCount,
                model_id: SAM2_TRACKER_MODEL_ID,
                video_name: jobInstance.taskName || `Task ${jobInstance.taskId}`,
            });
            let semanticGroupID = objectState.group?.id ?? null;
            if (actionInstance instanceof Task) {
                const prediction = (await actionInstance.annotations.get(firstPredictedFrame, false, []))
                    .find((state) => state.clientID === actionObjectState.clientID);
                semanticGroupID = prediction?.group?.id ?? semanticGroupID;
                await actionInstance.annotations.save(undefined, true);
            }
            if (playing) {
                onSwitchPlay(false);
            }
            if (destinationJob.id === jobInstance.id) {
                if (actionInstance instanceof Task) {
                    await jobInstance.annotations.clear({ reload: true });
                }
                this.changeFrame(firstPredictedFrame);
                activateSAM2Prediction(objectState.clientID);
            } else {
                writeLatestFrame(destinationJob.id, firstPredictedFrame);
                history.push(
                    `/tasks/${jobInstance.taskId}/jobs/${destinationJob.id}?frame=${firstPredictedFrame}`,
                    {
                        sam2Prediction: {
                            groupID: semanticGroupID,
                            labelID: objectState.label.id,
                        },
                    } satisfies SAM2NavigationState,
                );
            }
            notification.success({
                key: progressKey,
                message: `SAM2 predictions ready ${directionLabel}`,
                description: `Opened frame ${firstPredictedFrame}`,
                duration: 2,
                placement: 'bottomRight',
            });
        } catch (error) {
            if (trackingAttempted) {
                jobInstance.logger.log(EventScope.sam2Tracking, {
                    duration: Math.round(performance.now() - trackingStartedAt),
                    outcome: 'failed',
                    direction: directionLabel,
                    requested_frames: sam2FrameCount,
                    model_id: SAM2_TRACKER_MODEL_ID,
                    video_name: jobInstance.taskName || `Task ${jobInstance.taskId}`,
                    error_type: error instanceof Error ? error.name : 'unknown',
                });
            }
            notification.error({
                key: progressKey,
                message: error instanceof Error ? error.message : String(error),
            });
        } finally {
            this.trackingSAM2 = false;
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
        } = this.props;
        const sam2KeyMap = workspace === Workspace.STANDARD ? subKeyMap(componentShortcuts, keyMap) : {};

        const topBar = (
            <AnnotationTopBarComponent
                showStatistics={this.showStatistics}
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
                    }}
                />
                {topBar}
            </>
        );
    }
}

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(AnnotationTopBarContainer));
