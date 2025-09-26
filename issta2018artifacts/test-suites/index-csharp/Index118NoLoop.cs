using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


public class Index118NoLoop {

    public static void foo(String [] args, int i) {
        Contract.Requires(args.Length == 4);
        if (i >= 1 && i <= 3) {
            Console.WriteLine(args[i]);
        }
    }
}
