using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Simplified test case for bug fixed by Daikon commit f123f498b

class File
{
    public File(string v) { }
}

class CommandLine
{
    private static bool dump_dtrace;

    /** Reads a decl file and dumps statistics. */
    public static void Main(string[] args)
    {
        string[] files = getFilesFromOptions(args);

        // If reading/dumping dtrace file, just read one file and dump it
        if (dump_dtrace)
        {
            DTraceReader trace = new DTraceReader();
            //:: error: array.access.unsafe.high.constant
            trace.read(new File(files[0]));
            trace.dump_data();
            return;
        }

        // several more nearly identical examples of the same pattern.
    }

    /* In the actual example, there was real argument parsing here,
    * which also sets various flags, like dump_dtrace.
    */
    public static string[] getFilesFromOptions(string[] args)
    {
        return null;
    }

    class DTraceReader
    {
        public void read(File f) { }
        public void dump_data() { }
    }
}

