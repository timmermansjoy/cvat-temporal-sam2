// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

// Run from the repository root: node cvat-ui/tests/sam2-cancellation.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const lodash = require('lodash');

function load(relativePath, imports) {
    const filename = path.resolve(__dirname, relativePath);
    const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const exports = {};
    new Function('require', 'exports', compiled)((name) => {
        assert.ok(name in imports, `Unexpected import: ${name}`);
        return imports[name];
    }, exports);
    return exports;
}

class Job {}
class Task {}
Job.prototype.frames = { get: { implementation: async (number) => ({ number, width: 100, height: 100 }) } };
const ObjectType = { SHAPE: 'shape', TRACK: 'track', TAG: 'tag' };
const base = load('../../cvat-core/src/annotations-actions/base-action.ts', {});
const runner = load('../../cvat-core/src/annotations-actions/base-collection-action.ts', {
    lodash,
    '../session': { Job, Task },
    '../enums': { EventScope: {}, ObjectType },
    '../annotations': { getCollection: (job) => ({ export: () => job.collection }) },
    '../annotations-filter': { default: class {
        filterSerializedCollection = (collection) => Object.fromEntries(
            Object.entries(collection).map(([kind, objects]) => [kind, objects.map(({ clientID }) => clientID)]),
        );
    } },
    './base-action': base,
});
const core = { lambda: {} };
const Tracker = load('../src/utils/annotations-actions/sam2-tracker.ts', {
    lodash,
    'cvat-logger': { EventScope: {} },
    'cvat-core-wrapper': {
        ...base, ...runner, Job, Task, ObjectType,
        ShapeType: { MASK: 'mask', POLYGON: 'polygon' }, Source: { AUTO: 'auto' }, getCore: () => core,
    },
}).default;

async function check() {
    const points = [1, 0, 0, 0, 0];
    const track = {
        clientID: 1, frame: 0, label_id: 1, group: 0, attributes: [], elements: [],
        shapes: [0, 11].map((frame) => ({ frame, type: 'mask', points, source: 'manual', outside: false })),
    };
    const collection = { shapes: [], tags: [], tracks: [track] };
    const original = structuredClone(collection);
    const state = {
        clientID: 1, shapeType: 'mask', objectType: 'track', points,
        export: async () => structuredClone(track),
    };
    let commits = 0;
    let cancelled = false;
    let lastProgress;
    const job = Object.assign(new Job(), {
        id: 1, taskId: 1, stopFrame: 100, labels: [], collection,
        frames: {
            frameNumbers: async () => Array.from({ length: 101 }, (_, frame) => frame),
            get: async () => ({ deleted: false }),
        },
        annotations: {
            get: async () => [state],
            commit: async (created) => {
                commits++;
                assert.equal(lastProgress, 100, 'Final progress must flush before committing');
                assert.ok(created.tracks[0].shapes.some(({ frame }) => frame === 100));
            },
        },
        logger: { log: async () => ({ close: () => {} }) },
    });
    const params = { 'Target frame': '100', 'Frame count': '100', 'Convert polygon shapes to tracks': 'false' };
    const scenarios = ['pth-facebookresearch-sam2', 'pth-facebookresearch-sam3-1']
        .flatMap((modelID) => ['initialize', 'second batch', 'last batch', null]
            .map((cancelAt) => ({ modelID, cancelAt })));
    for (const { modelID, cancelAt } of scenarios) {
        cancelled = false;
        commits = 0;
        const requests = [];
        core.lambda.call = async (_task, _model, request) => {
            requests.push(request);
            await Promise.resolve();
            if (request.type === 'init_tracking') {
                if (cancelAt === 'initialize') cancelled = true;
                return { states: ['seed'] };
            }
            if ((cancelAt === 'second batch' && request.frames.includes(11)) ||
                (cancelAt === 'last batch' && request.frames.includes(100))) {
                cancelled = true;
            }
            return {
                states: ['next'],
                frame_results: request.frames.map(() => [{ type: 'mask', points }]),
            };
        };
        const tracker = new Tracker({ id: modelID, version: 3 });
        assert.equal(tracker.modelID, modelID);
        assert.equal(tracker.name, modelID.endsWith('sam3-1') ?
            'Segment Anything 3.1: Tracker' : 'Segment Anything 2: Tracker');
        const progress = [];
        await runner.call(job, tracker, params, 0, [state], (_message, percent) => {
            lastProgress = percent;
            progress.push(percent);
        }, () => cancelled);
        assert.equal(commits, cancelAt ? 0 : 1, `Unexpected annotation commit: ${cancelAt}`);
        assert.deepEqual(collection, original, 'Cancelled work must not mutate existing annotations');
        if (cancelAt === 'initialize') assert.equal(requests.length, 1);
        if (cancelAt === 'second batch') {
            assert.equal(requests.length, 3, 'No correction request or next batch may start after cancellation');
            assert.deepEqual(requests[2].frames, [11]);
        }
        const progressCount = progress.length;
        await new Promise((resolve) => setTimeout(resolve, 120));
        assert.equal(progress.length, progressCount, 'Progress must stop when the action returns');
    }

    // The shared runner must discard a late result even if an action ignores cancellation.
    for (const method of ['call', 'run']) {
        cancelled = false;
        commits = 0;
        const action = Object.assign(new runner.BaseCollectionAction(), {
            name: 'late result', parameters: null,
            init: async () => {}, destroy: async () => {}, applyFilter: (input) => input.collection,
            run: async () => {
                cancelled = true;
                return { created: collection, deleted: collection };
            },
        });
        await runner[method](job, action, {}, 0, method === 'call' ? [state] : [], () => {}, () => cancelled);
        assert.equal(commits, 0, `${method} must discard cancelled results before committing`);
    }
    console.log('SAM2 and SAM3.1 cancellation checks passed');
}

check().catch((error) => { console.error(error); process.exitCode = 1; });
