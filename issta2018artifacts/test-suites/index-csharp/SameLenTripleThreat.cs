using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class SameLenTripleThreat {
    public void foo(String[] vars) {
        String[] qrets = new String[vars.Length];
        String[] y = qrets;
        if(TestHelper.nondet()) Contract.Assert(y.Length == vars.Length);
        String[] indices = new String[vars.Length];
        String[] x = indices;
        if(TestHelper.nondet()) Contract.Assert(x.Length == qrets.Length);
    }

    String[] indices;

    public void foo2(params String[] vars) {
        String[] qrets = new String[vars.Length];
        indices = new String[vars.Length];
        String[] indicesLocal = new String[vars.Length];
        for (int i = 0; i < qrets.Length; i++) {
            indices[i] = "hello";
            indicesLocal[i] = "hello";
        }
    }

    public void foo3(params String[] vars) {
        String[] qrets = new String[vars.Length];
        String[] indicesLocal = new String[vars.Length];
        indices = new String[vars.Length];
        for (int i = 0; i < qrets.Length; i++) {
            indices[i] = "hello";
            indicesLocal[i] = "hello";
        }
    }
}
