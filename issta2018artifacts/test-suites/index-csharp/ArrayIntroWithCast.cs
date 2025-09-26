using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ArrayIntroWithCast<T> where T:class{

    void test(String[] a, String[] b) {
        Array result = new Object[a.Length + b.Length];
        Array.Copy(a, 0, result, 0, a.Length);
    }

    void test2(String[] a, String[] b) {
        /*@SuppressWarnings("unchecked")*/
        T[] result = (T[]) new Object[a.Length + b.Length];
        Array.Copy(a, 0, result, 0, a.Length);
    }
}
