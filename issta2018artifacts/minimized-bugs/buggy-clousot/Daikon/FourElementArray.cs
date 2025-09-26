using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class FourElementArray
{
    /* Every null in the function below was originally some meaningful String.
     * The original documentation specified that this function always returns
     * an array with four elements.
     */
    public static string[] esc_quantify(params object[] vars)
    {
        Contract.Ensures(Contract.Result<string[]>().Length >= 4);

        if (vars.Length == 1)
        {
            //:: error: return.type.incompatible
            return new string[] { null, null, ")" };
        }
        else if ((vars.Length == 2) && boolean_condition(vars[1]))
        {
            return new string[] {
                null,
                null,
                null,
                ")"
            };
        }
        else
        {
            return new string[] {
                null, null, null, ")"
            };
        }
    }

    /* In the actual example this was something meaningful. */
    private static bool boolean_condition(object o)
    {
        return true;
    }
}
