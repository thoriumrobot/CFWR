using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Arrays
{
    public static void Fill<T>(T[] a, T v)
    {
        for (int i = 0; i < a.Length; ++i)
            a[i] = v;
    }
}

public class FnInverse
{

    public static int[] fn_inverse(int[] a, int arange)
    {
        Contract.Requires(arange >= 0);
        int[] result = new int[arange];
        Arrays.Fill(result, -1);
        for (int i = 0; i < a.Length; i++)
        {
            int ai = a[i];
            if (ai != -1)
            {
                //:: error: array.access.unsafe.high error: array.access.unsafe.low
                if (result[ai] != -1)
                {
                    throw new InvalidOperationException("Not invertible");
                }
                //:: error: array.access.unsafe.high error: array.access.unsafe.low
                result[ai] = i;
            }
        }
        return result;
    }
}