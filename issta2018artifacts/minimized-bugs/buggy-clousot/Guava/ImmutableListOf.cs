using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// In older .NET, maximum array size used to be 2 GB, now using some config file you should be able to get more,
// but still the maximum number of elements is int.MaxValue. 
// Therefore, it should be the same situation as in Java.
//
// Some related links:
// https://blogs.msdn.microsoft.com/joshwil/2005/08/10/bigarrayt-getting-around-the-2gb-array-size-limit/
// https://docs.microsoft.com/en-us/dotnet/framework/configure-apps/file-schema/runtime/gcallowverylargeobjects-element


public class ImmutableListOf
{
    /**
     * Returns an immutable set containing the given elements, minus duplicates, in the order each was
     * first specified. That is, if multiple elements are {@linkplain Object#equals equal}, all except
     * the first are ignored.
     *
     * @since 3.0 (source-compatible since 2.0)
     */
    
    public static void of<E>(E e1, E e2, E e3, E e4, E e5, E e6, params E[] others) where E : class
    {
        int paramCount = 6;
        object[] elements = new object[paramCount + others.Length];
        // Note that all of these are found to be unsafe because the value checker can't prove the minlen of elements, because paramCount + others.length might overflow
        //:: error: (array.access.unsafe.high.constant)
        elements[0] = e1;
        //:: error: (array.access.unsafe.high.constant)
        elements[1] = e2;
        //:: error: (array.access.unsafe.high.constant)
        elements[2] = e3;
        //:: error: (array.access.unsafe.high.constant)
        elements[3] = e4;
        //:: error: (array.access.unsafe.high.constant)
        elements[4] = e5;
        //:: error: (array.access.unsafe.high.constant)
        elements[5] = e6;
        Array.Copy(others, 0, elements, paramCount, others.Length);
    }
}