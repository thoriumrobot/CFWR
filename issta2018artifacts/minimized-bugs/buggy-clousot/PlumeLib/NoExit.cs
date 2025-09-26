using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// code that causes the bug fixed in plume-lib by rev. 95b3cab

public class NoExit
{
    public static void Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("Needs 2 arguments, got " + args.Length);
        }
        //:: error: (array.access.unsafe.high.constant)
        int limit = int.Parse(args[0]);
    }
}