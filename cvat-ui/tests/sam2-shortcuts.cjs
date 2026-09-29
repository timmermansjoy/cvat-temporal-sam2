// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

// Run from the repository root: node cvat-ui/tests/sam2-shortcuts.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');

class Job {}
class BaseCollectionAction {
    name = 'Segment Anything 2: Tracker';
    isApplicableForObject = (state) => ['mask', 'polygon'].includes(state.shapeType);
}

const calls = [];
const notices = [];
const activeNotices = new Map();
let canNavigate = true;
let failTracking = false;
let shortcuts;
const core = {
    actions: {
        list: async () => [
            new BaseCollectionAction(),
            Object.assign(new BaseCollectionAction(), { name: 'Segment Anything 3.1: Tracker' }),
        ],
        call: async (...args) => {
            calls.push(args);
            if (failTracking) throw new Error('Tracking failed');
            args[5]('Tracking with SAM2', 100);
            assert.equal(args[6](), false, 'Closing completed progress must not cancel the batch');
        },
    },
    tasks: { get: async () => { assert.fail('Propagation must stay in the current job'); } },
};
const mocks = {
    react: { default: React },
    'react-redux': { connect: () => (component) => component },
    'react-router': { withRouter: (component) => component },
    'antd/lib/notification': { default: Object.fromEntries(
        ['info', 'warning', 'success', 'error', 'destroy'].map((method) => [
            method, (notice) => {
                notices.push([method, notice]);
                if (method === 'destroy') {
                    activeNotices.get(notice)?.onClose?.();
                    activeNotices.delete(notice);
                } else {
                    activeNotices.set(notice.key, notice);
                }
            },
        ]),
    ) },
    'antd/lib/progress': { default: 'progress' },
    'cvat-logger': { EventScope: { sam2Tracking: 'tracking' } },
    'actions/annotation-actions': {},
    'actions/shortcuts-actions': { registerComponentShortcuts: (value) => { shortcuts = value; } },
    'components/annotation-page/top-bar/top-bar': { default: 'top-bar' },
    'cvat-core-wrapper': {
        Job, BaseCollectionAction, getCore: () => core,
        ShapeType: { MASK: 'mask', POLYGON: 'polygon' },
        Source: { AUTO: 'auto' }, JobType: { ANNOTATION: 'annotation' },
    },
    reducers: { Workspace: { STANDARD: 'standard' } },
    'utils/is-able-to-change-frame': { default: () => canNavigate },
    'utils/mousetrap-react': { default: 'hotkeys' },
    'actions/settings-actions': {},
    'utils/remember-latest-frame': { writeLatestFrame: () => {} },
    'utils/drawing': {},
    'utils/to-clipboard': {},
    'utils/annotations-actions/sam2-tracker': {
        SAM2_TRACKER_ACTION_NAME: 'Segment Anything 2: Tracker',
        SAM2_TRACKER_MODEL_ID: 'pth-facebookresearch-sam2',
        SAM31_TRACKER_ACTION_NAME: 'Segment Anything 3.1: Tracker',
        SAM31_TRACKER_MODEL_ID: 'pth-facebookresearch-sam3-1',
    },
    'utils/enums': { ShortcutScope: { STANDARD_WORKSPACE: 'standard' } },
    'utils/component-subkeymap': { subKeyMap: (value) => value },
};

function load(relativePath, imports) {
    const filename = path.join(__dirname, relativePath);
    const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
    }).outputText;
    const exports = {};
    new Function('require', 'exports', compiled)((name) => {
        assert.ok(name in imports, `Unexpected import: ${name}`);
        return imports[name];
    }, exports);
    return exports;
}
const TopBar = load('../src/containers/annotation-page/top-bar/top-bar.tsx', mocks).default;
const TopBarComponent = load('../src/components/annotation-page/top-bar/top-bar.tsx', {
    react: { default: React },
    'antd/lib/grid': { Col: 'col', Row: 'row' },
    'antd/lib/select': { default: 'select' },
    'components/common/cvat-tooltip': { default: 'tooltip' },
    reducers: mocks.reducers,
    ...Object.fromEntries(['left-group', 'player-buttons', 'player-navigation', 'right-group']
        .map((name) => [`./${name}`, { default: name }])),
}).default;

function findSelect(element, className = 'cvat-sam2-frame-count') {
    if (element?.type === 'select' && element.props.className === className) return element;
    return React.Children.toArray(element?.props?.children).map((child) => findSelect(child, className)).find(Boolean);
}

const object = (clientID, extra = {}) => ({
    clientID, serverID: clientID + 100, frame: 0, updated: clientID,
    shapeType: 'mask', objectType: 'track', label: { id: 1 }, group: { id: 0 },
    outside: false, hidden: false, lock: false, ...extra,
});
const mask = object(1);
const polygon = object(2, { shapeType: 'polygon', objectType: 'shape' });
const states = [
    mask, polygon, object(3, { lock: true }), object(4, { hidden: true }),
    object(5, { outside: true }), object(6, { shapeType: 'rectangle' }), object(7, { frame: 1 }),
];

async function check() {
    let chosenFrameCount;
    const frameCountProps = {
        workspace: 'standard', sam2FrameCount: 7,
        onChangeSAM2FrameCount: (value) => { chosenFrameCount = value; },
        samTrackerModelID: 'pth-facebookresearch-sam2',
        samTrackerModels: [{ value: 'pth-facebookresearch-sam3-1', label: 'SAM3.1' }],
        onChangeSAMTrackerModel: (value) => { frameCountProps.samTrackerModelID = value; },
    };
    const frameCountSelect = findSelect(TopBarComponent(frameCountProps));
    assert.equal(frameCountSelect.props.value, 7, 'Keep custom frame counts from Settings visible');
    assert.deepEqual(frameCountSelect.props.options.map(({ value }) => value), [1, 5, 7, 10, 25, 50, 100]);
    frameCountSelect.props.onChange(25);
    assert.equal(chosenFrameCount, 25);
    assert.equal(findSelect(TopBarComponent({ ...frameCountProps, workspace: 'review' })), undefined);
    findSelect(TopBarComponent(frameCountProps), 'cvat-sam-tracker-model')
        .props.onChange('pth-facebookresearch-sam3-1');
    assert.equal(frameCountProps.samTrackerModelID, 'pth-facebookresearch-sam3-1');

    let openedFrame;
    let expectedFrame = 1;
    let selected;
    const job = Object.assign(new Job(), {
        id: 10, taskId: 20, startFrame: 0, stopFrame: 5, type: 'annotation', parentJobId: null,
        frames: {
            frameNumbers: async () => [0, 1, 2, 3, 4, 5],
            get: async () => ({ deleted: false }),
            search: async () => expectedFrame,
        },
        annotations: { save: async () => {}, get: async () => states },
        logger: { log: () => {} },
    });
    const props = {
        jobInstance: job, objectStates: states, frameNumber: 0, sam2FrameCount: 5,
        activatedStateID: mask.clientID, normalizedKeyMap: {}, workspace: 'standard',
        onChangeFrame: async (frame) => { await Promise.resolve(); openedFrame = frame; },
        activateSAM2Prediction: (id) => {
            assert.equal(openedFrame, expectedFrame, 'Select the object after navigation finishes');
            selected = id;
        },
        history: { push: () => assert.fail('Propagation must not change jobs') },
    };
    const topBar = new TopBar(props);
    assert.deepEqual(shortcuts.SAM2_TRACK_ALL_FORWARD.sequences, ['shift+g']);
    assert.deepEqual(shortcuts.SAM2_TRACK_ALL_BACKWARD.sequences, ['shift+s']);
    const hotkeys = topBar.render().props.children[0];
    const originalTracking = topBar.onTrackSAM2;
    topBar.onTrackSAM2 = (...args) => assert.deepEqual(args, [1, true]);
    hotkeys.props.handlers.SAM2_TRACK_ALL_FORWARD();
    topBar.onTrackSAM2 = (...args) => assert.deepEqual(args, [-1, true]);
    hotkeys.props.handlers.SAM2_TRACK_ALL_BACKWARD();
    topBar.onTrackSAM2 = originalTracking;

    await topBar.onTrackSAM2(1, true);
    assert.equal(calls.length, 1, 'All objects must share one action/commit');
    assert.deepEqual(calls[0][4].map((state) => state.clientID).sort(), [1, 2]);
    assert.equal(calls[0][2]['Frame count'], '5');
    assert.equal(selected, mask.clientID);

    calls.length = 0;
    await topBar.onTrackSAM2(1);
    assert.deepEqual(calls[0][4], [mask], 'G must still propagate only the selected object');
    calls.length = 0;
    topBar.props = { ...props, activatedStateID: null };
    await topBar.onTrackSAM2(1, true);
    assert.equal(calls[0][4].length, 2, 'Batch propagation requires no selected object');

    calls.length = 0;
    expectedFrame = 4;
    topBar.props = {
        ...props, frameNumber: 5,
        objectStates: states.map((state) => ({ ...state, frame: state.frame + 5 })),
    };
    await topBar.onTrackSAM2(-1, true);
    assert.equal(calls.length, 1, 'Backward propagation must use one batch');
    assert.deepEqual(calls[0][4].map((state) => state.clientID).sort(), [1, 2]);
    assert.equal(calls[0][3], 5);
    assert.equal(calls[0][2]['Target frame'], '0');
    assert.equal(calls[0][2]['Frame count'], '5');
    assert.equal(openedFrame, 4, 'Open the first backward prediction');
    assert.equal(selected, mask.clientID);
    calls.length = 0;
    await topBar.onTrackSAM2(-1);
    assert.deepEqual(calls[0][4].map((state) => state.clientID), [1], 'S still tracks only the selected object');
    expectedFrame = 1;

    for (const unavailable of [{ frameIsDeleted: true }, { frameFetching: true }, { objectStates: [] }]) {
        calls.length = 0;
        topBar.props = { ...props, ...unavailable };
        await topBar.onTrackSAM2(1, true);
        assert.equal(calls.length, 0);
    }
    topBar.props = props;
    canNavigate = false;
    await topBar.onTrackSAM2(1, true);
    assert.equal(calls.length, 0, 'Do not track during drawing or another canvas operation');
    canNavigate = true;
    failTracking = true;
    openedFrame = undefined;
    await topBar.onTrackSAM2(1, true);
    assert.equal(openedFrame, undefined, 'Failed tracking must not navigate');
    assert.equal(topBar.trackingSAM2, false, 'A failed batch can be retried');
    failTracking = false;

    const closeProgress = () => {
        const progress = [...activeNotices.values()].find((notice) => notice.duration === 0);
        assert.ok(progress?.onClose, 'Progress must offer cancellation');
        progress.onClose();
        activeNotices.delete(progress.key);
    };
    calls.length = 0;
    const preparing = topBar.onTrackSAM2(1, true);
    closeProgress();
    await preparing;
    assert.equal(calls.length, 0, 'Closing during preparation must prevent tracking');

    const originalCall = core.actions.call;
    const cancelDuringInference = async (...args) => {
        calls.push(args);
        args[5]('Tracking frame 1 of 100', 1);
        closeProgress();
        assert.equal(args[6](), true, 'The tracker must see cancellation');
        const cancelling = [...activeNotices.values()].find((notice) => notice.message === 'Cancelling SAM2 tracking…');
        assert.equal(cancelling?.duration, 0, 'Cancellation feedback must remain until the request finishes');
        assert.equal(cancelling.closable, false);
        const noticeCount = notices.length;
        args[5]('Late progress', 20);
        args[5]('Tracking with SAM2', 100);
        assert.equal(notices.length, noticeCount, 'Cancelled progress must not reappear');
        await topBar.onTrackSAM2(1, true);
        assert.equal(calls.length, 1, 'Wait for the pending request before allowing another batch');
    };
    core.actions.call = cancelDuringInference;
    for (const direction of [-1, 1]) {
        expectedFrame = direction === 1 ? 1 : 4;
        topBar.props = direction === 1 ? props : {
            ...props, frameNumber: 5,
            objectStates: states.map((state) => ({ ...state, frame: state.frame + 5 })),
        };
        for (const allObjects of [false, true]) {
            calls.length = 0;
            notices.length = 0;
            openedFrame = undefined;
            await topBar.onTrackSAM2(direction, allObjects);
            assert.equal(openedFrame, undefined, 'Cancelled tracking must keep the current frame');
            assert.equal(topBar.trackingSAM2, false, 'Cancellation must allow a retry after the request finishes');
            assert.ok(notices.every(([kind]) => kind !== 'success' && kind !== 'error'));
            const cancelledNotice = [...activeNotices.values()]
                .find((notice) => notice.message === 'SAM2 tracking cancelled');
            assert.equal(cancelledNotice?.duration, 3, 'Replace waiting feedback when retry becomes available');
        }
    }
    core.actions.call = async (...args) => {
        await cancelDuringInference(...args);
        throw new Error('Request failed after cancellation');
    };
    calls.length = 0;
    await topBar.onTrackSAM2(1, true);
    assert.ok(notices.every(([kind]) => kind !== 'error'), 'Cancellation must suppress late request errors');
    core.actions.call = originalCall;
    await topBar.onTrackSAM2(1, true);
    assert.equal(openedFrame, 1, 'Tracking must work again after cancellation');

    calls.length = 0;
    topBar.props = { ...props, samTrackerModelID: 'pth-facebookresearch-sam3-1' };
    await topBar.onTrackSAM2(1, true);
    assert.equal(calls[0][1].name, 'Segment Anything 3.1: Tracker', 'Shortcuts must use the selected model');
    assert.ok(notices.some(([kind, notice]) => kind === 'success' && notice.message.startsWith('SAM3.1')));
    topBar.props = props;

    calls.length = 0;
    job.stopFrame = 2;
    await topBar.onTrackSAM2(1, true);
    assert.equal(calls[0][0], job);
    assert.equal(calls[0][2]['Target frame'], '2', 'Long runs must stop at the job boundary');
    job.stopFrame = 0;
    for (const direction of [-1, 1]) {
        calls.length = 0;
        await topBar.onTrackSAM2(direction, true);
        assert.equal(calls.length, 0, 'Do not propagate beyond the first or last job frame');
    }
    console.log('SAM2 shortcut checks passed');
}

check().catch((error) => { console.error(error); process.exitCode = 1; });
