using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Simplified test case for Daikon commit 29dfd8dde

class GetSimplifyFreeIndices
{
    public static object get_simplify_free_indices(params object[] vars)
    {
        if (vars.Length == 1) {
            return vars[0];
        }
        else if(vars.Length == 2)
        {
            //:: error: array.access.unsafe.high.constant
            return compose(vars[0], vars[2]);
        }
        else
        {
            throw new Exception("unexpected length" + vars.Length);
        }
    }
    public static object compose(object ob1, object obj2)
    {
        return null;
    }
}
