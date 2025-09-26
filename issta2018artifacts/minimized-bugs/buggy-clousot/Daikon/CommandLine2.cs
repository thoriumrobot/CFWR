using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


class CommandLine2
{
    private static bool primitive_declaration_type_comparability;

    /** Reads a decl file and dumps statistics. */
    public static void Main(string[] args)
    {
        string[] files = getFilesFromOptions(args);

        // If reading/dumping dtrace file, just read one file and dump it
        if (primitive_declaration_type_comparability)
        {
            if (files.Length != 1)
            {
                Console.Error.WriteLine("One decl-file expected, received " + files.Length + ".");
                Environment.Exit(1);
            }
            DeclReader dr = new DeclReader();
            dr.read(new File(files[0]));
            dr.primitive_declaration_types();
            //:: error: array.access.unsafe.high.constant
            dr.write_decl(files[1]);
            return;
        }
    }

    /* In the actual example, there was real argument parsing here,
     * which also sets various flags, like 
     * primitive_declaration_type_comparability.
     */
    public static string[] getFilesFromOptions(string[] args)
    {
        return null;
    }

    class DeclReader
    {
        public void read(File f) { }
        public void primitive_declaration_types() { }
        public void write_decl(string decl) { }
    }
}

