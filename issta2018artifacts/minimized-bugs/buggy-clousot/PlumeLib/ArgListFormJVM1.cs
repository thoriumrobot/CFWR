using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// simplified version of the first of the two bugs fixed in plume-lib rev. 4f10607
// disabled (actually not a bug)

#if false

public class ArgListFromJVM1
{
    public static void arglistFromJvm(string arglist)
    {
        if (!(arglist.StartsWith("(", StringComparison.Ordinal) && arglist.EndsWith(")", StringComparison.Ordinal)))
        {
            throw new Exception("Malformed arglist: " + arglist);
        }
        string result = "(";
        int pos = 1;
        while (pos < arglist.Length - 1)
        {
            if (pos > 1)
            {
                result += ", ";
            }
            int nonarray_pos = pos;
            //:: error: (argument.type.incompatible)
            while (arglist[nonarray_pos] == '[')
            {
                nonarray_pos++;
            }
        }
    }
}
#endif