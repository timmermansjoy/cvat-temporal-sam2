// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import type ObjectState from '../object-state';
import { ShapeType } from '../enums';
import type { SerializedTrack } from '../server-response-types';
import { checkNumberOfPoints, cropMask } from '../object-utils';
import type { AnnotationInjection, InterpolatedPosition } from './types';
import { MaskShape } from './mask-shape';
import { Track } from './track';

export class MaskTrack extends Track {
    constructor(data: SerializedTrack, clientID: number, color: string, injection: AnnotationInjection) {
        super(data, clientID, color, injection);
        this.shapeType = ShapeType.MASK;
        for (const [frame, shape] of Object.entries(this.shapes)) {
            checkNumberOfPoints(this.shapeType, shape.points);
            const { width, height } = this.framesInfo[+frame];
            shape.points = cropMask(shape.points, width, height);
            shape.rotation = 0;
        }
    }

    protected interpolatePosition(leftPosition, _rightPosition, _offset): InterpolatedPosition {
        return {
            points: [...leftPosition.points],
            rotation: 0,
            occluded: leftPosition.occluded,
            outside: leftPosition.outside,
            zOrder: leftPosition.zOrder,
        };
    }

    protected validateStateBeforeSave(
        data: ObjectState,
        updated: ObjectState['updateFlags'],
        frame?: number,
    ): number[] {
        super.validateStateBeforeSave(data, updated, frame);
        if (updated.points) {
            const { width, height } = this.framesInfo[frame];
            return cropMask(data.points, width, height);
        }

        return [];
    }
}

Object.defineProperty(MaskTrack, 'distance', { value: MaskShape.distance });
