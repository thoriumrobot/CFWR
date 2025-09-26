using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// a simplified version of the bug fixed by plume-lib revision 7a6f75bf5a8b72a6ae823d4bd29a43742eb5cf50

class StackTraceElement
{
    public string getClassName() { return null; }
    public string getFileName() { return null; }
    public string getMethodName() { return null; }
    public int getLineNumber() { return 0; }
}

class Throwable
{
    public void fillInStackTrace() { }
    public StackTraceElement[] getStackTrace() { return null; }
}

public class GetStackTrace
{
    public void error()
    {
        Throwable t = new Throwable();
        t.fillInStackTrace();
        StackTraceElement[] ste = t.getStackTrace();
        //:: error: array.access.unsafe.high.constant
        StackTraceElement caller = ste[1];
        Console.Write(
                  "{0}.{1} ({2} line {3})",
                  caller.getClassName(),
                  caller.getMethodName(),
                  caller.getFileName(),
                  caller.getLineNumber());
        for (int ii = 2; ii < ste.Length; ii++)
        {
            Console.Write(" [{0} line {1}]", ste[ii].getFileName(), ste[ii].getLineNumber());
        }
    }
}