using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// pre-bugfix code from plume-lib, fixed in f28b4a0ac4a7c683b7a7ecaeb9f6e734c5856c86

public class Sum
{
    public static int sum(int[][] a)
    {
        int sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a[0].Length; j++)
            {
                //:: error: array.access.unsafe.high
                // a[0].length == a[i].length not true for all arrays!
                sum += a[i][j];
            }
        }
        return sum;
    }
}