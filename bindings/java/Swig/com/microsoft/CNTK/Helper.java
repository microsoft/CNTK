//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Helper.java -- Common helper functions, used only.
//
package com.microsoft.CNTK;

import java.util.List;

class Helper {
        public static FloatVector AsFloatVector(float[] input) {
            FloatVector inputVector = new FloatVector();
            for (float element: input) {
                inputVector.add(element);
            }
            return inputVector;
        }

        public static DoubleVector AsDoubleVector(double[] input) {
            DoubleVector inputVector = new DoubleVector();
            for (double element: input) {
                inputVector.add(element);
            }
            return inputVector;
        }

        public static SizeTVector AsSizeTVector(long[] input) {
            SizeTVector inputVector = new SizeTVector();
            for (long element: input) {
                inputVector.add(element);
            }
            return inputVector;
        }

        public static BoolVector AsBoolVector(boolean[] input) {
            BoolVector inputVector = new BoolVector();
            for (boolean element: input) {
                inputVector.add(element);
            }
            return inputVector;
        }
}
