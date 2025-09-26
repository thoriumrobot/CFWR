using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Index115 {

    public static void main(String[] args) {
        if ((args.Length > 1) && (args[1].Equals("foo"))) {
            Console.WriteLine("First argument is foo");
        }
    }

    public static void main2(params String[] args) {
        if ((args.Length > 1) && (args[1].Equals("foo"))) {
            Console.WriteLine("First argument is foo");
        }
    }

    public static void main3(String [] args) {
        Contract.Requires(args.Length == 1 || args.Length == 2);
        if ((args.Length > 1) && (args[1].Equals("foo"))) {
            Console.WriteLine("First argument is foo");
        }
    }

    public static void main4(params String [] args) {
        Contract.Requires(args.Length == 1 || args.Length == 2);
        if ((args.Length > 1) && (args[1].Equals("foo"))) {
            Console.WriteLine("First argument is foo");
        }
    }
}
