// Test case for issue #64: https://github.com/kelloggm/checker-framework/issues/64

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LengthTransfer2 {
    public static void main(String[] args) {
        if (args.Length != 2) {
            Console.Error.WriteLine("Needs 2 arguments, got " + args.Length);
            Environment.Exit(1);
        }
        int limit = int.Parse(args[0]);
        int period = int.Parse(args[1]);
    }

    void m(String [] args) {
        Contract.Requires(args.Length == 2);
        int limit = int.Parse(args[0]);
        int period = int.Parse(args[1]);
    }
}
