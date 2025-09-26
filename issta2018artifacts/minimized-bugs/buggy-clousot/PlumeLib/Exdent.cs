using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// version of code that causes bug fixed by plume-lib rev. 339602f

public class Exdent
{
    bool enabled = true;
    string indent_str;
    readonly string INDENT_STR_ONE_LEVEL = "    ";

    /** Exdents: reduces indentation and pops a start time. */
    public void exdent()
    {
        //:: error: (argument.type.incompatible)
        indent_str = indent_str.Substring(0, indent_str.Length - INDENT_STR_ONE_LEVEL.Length);
    }
}