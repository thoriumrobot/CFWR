// Test case for https://github.com/kelloggm/checker-framework/issues/154
// This class wraps an array, but doesn't expose the array in its public interface. This test
// ensures that indexes for this new collection can be annotated as if the collection were an array.

// Note that there is a copy of this code in the manual in index-checker.tex. If this code is
// updated, you MUST update that copy, as well.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

/** ArrayWrapper is a fixed-size generic collection. */
public class ArrayWrapper<T> where T:class {
    private readonly Object [] @delegate;

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(@delegate.Length == this.size());
    }

        /*@SuppressWarnings("index")*/ // constructor creates object of size @SameLen(this) by definition
        ArrayWrapper(int size) {
        Contract.Requires(size >= 0);
        @delegate = new Object[size];
    }

    [Pure]
    public int size() {
        return @delegate.Length;
    }

    public void set(int index, T obj) {
        Contract.Requires(index >= 0 && index<this.size());
        @delegate[index] = obj;
    }

    /*@SuppressWarnings("unchecked")*/ // required for normal Java compilation due to unchecked cast
    public T get(int index) {
         Contract.Requires(index >= 0 && index<this.size());
        return (T) @delegate[index];
    }

    public static void clearIndex1(ArrayWrapper<T> a, int i) {
        Contract.Requires(i >= 0 && i<a.size());
        a.set(i, null);
    }

    public static void clearIndex2(ArrayWrapper<T> a, int i) {
        if (0 <= i && i < a.size()) {
            a.set(i, null);
        }
    }

    public static void clearIndex3(ArrayWrapper<T> a, int i) {
        Contract.Requires(i >= 0);
        if (i < a.size()) {
            a.set(i, null);
        }
    }
}
