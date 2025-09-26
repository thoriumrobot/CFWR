using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class EnumValues {

    public enum Direction {
        NORTH,
        SOUTH,
        EAST,
        WEST
    };

    public static void enumValues() {
        Direction [] arr4 = (Direction[])Enum.GetValues(typeof(Direction)); /*VD: not sure if cast is safe*/
        if(TestHelper.nondet()) Contract.Assert(arr4.Length == 4);
        Direction[] arr = (Direction[])Enum.GetValues(typeof(Direction)); /*VD: not sure if cast is safe*/
        Direction a = arr[0];
        Direction b = arr[1];
        Direction c = arr[2];
        Direction d = arr[3];
        // :: error: (array.access.unsafe.high.constant)
        Direction e = arr[4];
    }
}
