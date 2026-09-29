// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

// Run from the repository root: node cvat-ui/tests/sam31-prompts.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');

const modelID = 'pth-facebookresearch-sam3-1';
const requests = [];
const created = [];
const notices = [];
const canvasCalls = [];
const shapes = [0.95, 0.7].map((score, index) => ({
    type: 'mask', points: [0, 4, index, 0, index + 1, 1],
    attributes: [{ spec_id: 0, value: String(score) }],
}));
let infer = async () => ({ shapes });
const core = {
    plugins: { register() {} },
    lambda: { call: async (...args) => { requests.push(args); return infer(); } },
    enums: { Source: { SEMI_AUTO: 'semi-auto' } },
    classes: { ObjectState: class { constructor(data) { Object.assign(this, data); } } },
};
const mocks = {
    react: { default: React },
    'react-dom': { default: {} },
    'react-redux': { connect: () => (component) => component },
    '@ant-design/icons': { default: 'icon', QuestionCircleOutlined: 'help-icon' },
    'antd/lib/grid': { Row: 'row', Col: 'col' },
    'antd/lib/notification': { default: {
        error: (value) => notices.push(value), warning: (value) => notices.push(value),
    } },
    'antd/lib/message': { default: { loading: () => () => {}, info: () => () => {} } },
    lodash: { default: require('lodash') },
    icons: {},
    'cvat-canvas-wrapper': {
        convertShapesForInteractor: (items, type, kind) => items
            .filter((item) => item.type === type && item.kind === kind).map((item) => item.point),
    },
    'cvat-core-wrapper': {
        getCore: () => core, ShapeType: { MASK: 'mask', POLYGON: 'polygon' },
        ObjectType: { SHAPE: 'shape' }, DimensionType: { DIMENSION_2D: '2d' },
    },
    'utils/opencv-wrapper/opencv-wrapper': { default: {
        isInitialized: true,
        getContoursFromStateSync: () => [[[0, 0], [1, 0], [1, 1]]],
    } },
    'utils/annotations-actions/sam2-tracker': { SAM31_TRACKER_MODEL_ID: modelID },
    reducers: {},
    'actions/annotation-actions': {},
    'actions/settings-actions': {},
    './handle-popover-visibility': { default: (component) => component },
};
for (const name of ['popover', 'select', 'button', 'input', 'modal', 'typography/Text', 'tabs', 'switch']) {
    mocks[`antd/lib/${name}`] = { default: name };
}
mocks['antd/lib/select'] = { default: Object.assign(() => null, { Option: 'option' }) };
for (const name of [
    'components/model-runner-modal/detector-runner', 'components/model-runner-modal/region-of-interest-input',
    'components/label-selector/label-selector', 'components/common/cvat-tooltip', 'components/common/cvat-markdown',
    'components/annotation-page/standard-workspace/controls-side-bar/approximation-accuracy',
    'components/annotation-page/standard-workspace/controls-side-bar/confidence-threshold', './interactor-tooltips',
]) mocks[name] = { default: 'component' };

const filename = path.join(__dirname,
    '../src/components/annotation-page/standard-workspace/controls-side-bar/tools-control.tsx');
const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
}).outputText;
const loaded = {};
new Function('require', 'exports', compiled)((name) => {
    assert.ok(name in mocks, `Unexpected import: ${name}`);
    return mocks[name];
}, loaded);
const { ToolsControlComponent } = loaded;
global.localStorage = { getItem: () => null };
global.window = { addEventListener() {}, removeEventListener() {} };

function find(element, className) {
    if (element?.props?.className === className) return element;
    return React.Children.toArray(element?.props?.children).map((child) => find(child, className)).find(Boolean);
}
const sam31 = {
    id: `${modelID}--interactor`, name: 'SAM3.1', version: 3,
    params: { canvas: { minPosVertices: 0, minNegVertices: 0, startWithBoxOptional: true } },
};
const sam2 = { ...sam31, id: 'pth-facebookresearch-sam2--interactor' };

function makeComponent() {
    const props = {
        interactors: [sam31, sam2], trackers: [], detectors: [], labels: [{ id: 1 }], states: [],
        jobInstance: { id: 8, taskId: 7, dimension: '2d' }, activeLabelID: 1,
        frame: 12, frameData: { width: 16, height: 16 }, curZOrder: 0,
        refinementTargetID: null, interactorExtras: [], isActivated: false,
        toolsBlockerState: { algorithmsLocked: false },
        canvasInstance: {
            cancel() {}, interact: (value) => canvasCalls.push(value),
        },
        createAnnotations: (objects) => created.push(objects),
    };
    const component = new ToolsControlComponent(props);
    component.setState = (update) => { component.state = { ...component.state, ...update }; };
    props.onInteractionStart = () => {
        const previousProps = component.props;
        component.props = { ...previousProps, isActivated: true };
        component.componentDidUpdate(previousProps, component.state);
    };
    return component;
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

(async () => {
    const component = makeComponent();
    find(component.renderInteractorBlock(), 'cvat-sam-text-prompt').props.onChange({ target: { value: '  red car  ' } });
    const button = find(component.renderInteractorBlock(), 'cvat-tools-interact-button');
    assert.equal(button.props.children, 'Find objects');
    button.props.onClick();
    await settle();
    assert.deepEqual(requests.at(-1)[2], {
        frame: 12, obj_bbox: [], pos_points: [], neg_points: [], text_prompts: ['red car'], type: 'interact', job: 8,
    });
    assert.equal(canvasCalls.at(-1).command, 'put_shapes');
    assert.equal(canvasCalls.at(-1).payload.shapes.length, 2);
    assert.equal(created.length, 0, 'Detection must remain a preview until accepted');
    assert.equal(component.state.showConfidenceControl, true);

    component.onInteraction({ detail: { shapes: [{ type: 'points', kind: 'positive', point: [3, 4] }] } });
    await settle();
    assert.deepEqual(requests.at(-1)[2].pos_points, [[3, 4]]);
    assert.deepEqual(requests.at(-1)[2].text_prompts, ['red car']);
    component.setState({ thresholdValue: 0.8 });
    await component.interactionListener({ detail: { shapes: [], finished: true } });
    assert.equal(created.length, 1);
    assert.equal(created[0].length, 1, 'Acceptance must respect the confidence filter');
    assert.equal(created[0][0].label.id, 1);
    assert.equal(created[0][0].frame, 12);
    assert.deepEqual(created[0][0].points, shapes[0].points);

    const cancelled = makeComponent();
    let resolveInference;
    infer = () => new Promise((resolve) => { resolveInference = resolve; });
    cancelled.setState({ textPrompt: 'person' });
    find(cancelled.renderInteractorBlock(), 'cvat-tools-interact-button').props.onClick();
    await settle();
    await cancelled.cancelListener();
    const count = canvasCalls.length;
    resolveInference({ shapes });
    await settle();
    assert.equal(canvasCalls.length, count, 'A cancelled request must never put masks back on the canvas');
    assert.equal(created.length, 1);
    assert.equal(cancelled.state.fetching, false);

    const legacy = makeComponent();
    legacy.setActiveInteractor(sam2.id);
    assert.equal(find(legacy.renderInteractorBlock(), 'cvat-sam-text-prompt'), undefined);
    assert.equal(find(legacy.renderInteractorBlock(), 'cvat-tools-interact-button').props.children, 'Create mask');
    const before = requests.length;
    find(legacy.renderInteractorBlock(), 'cvat-tools-interact-button').props.onClick();
    await settle();
    assert.equal(requests.length, before, 'SAM2 must still wait for a point or box');

    const refinement = makeComponent();
    refinement.props = {
        ...refinement.props, refinementTargetID: 42,
        states: [{ clientID: 42, shapeType: 'mask', lock: false }],
    };
    refinement.setState({ textPrompt: 'person' });
    assert.equal(refinement.activeTextPrompt, '', 'Refining one accepted mask must not run text detection');
    assert.equal(find(refinement.renderInteractorBlock(), 'cvat-sam-text-prompt'), undefined);
    find(refinement.renderInteractorBlock(), 'cvat-tools-interact-button').props.onClick();
    assert.equal(canvasCalls.at(-1).command, 'draw_points', 'Refinement must not run exemplar box detection');
    await settle();
    assert.equal(requests.length, before);
    assert.deepEqual(notices, []);
    console.log('SAM3.1 text preview, point requests, acceptance, cancellation and SAM2 compatibility checks passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
