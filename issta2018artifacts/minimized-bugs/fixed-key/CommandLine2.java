// corrected test case for Daikon bug fixed in 13775fb6e

public class CommandLine2 {
    private static boolean primitive_declaration_type_comparability;

	            /*@ public normal_behavior
  @ requires true;
  @ ensures true;
  @*/
    /** Reads a decl file and dumps statistics. */
    public static void main(String[] args) {
	String[] files = getFilesFromOptions(args);
     
	// If reading/dumping dtrace file, just read one file and dump it
	if (primitive_declaration_type_comparability) {
	    if (files.length != 1) {
		println("One decl-file expected, received " + files.length + ".");
		return;
	    }
	    DeclReader dr = new DeclReader();
	    dr.read(new File(files[0]));
	    dr.primitive_declaration_types();
	    //:: error: array.access.unsafe.high.constant
	    dr.write_decl(files[0]);
	    return;
	}
    }

	/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    static void println(String s) { }

    /* In the actual example, there was real argument parsing here,
     * which also sets various flags, like 
     * primitive_declaration_type_comparability.
     */
                /*@ public normal_behavior
  @ requires true;
  @ ensures true;
  @*/
    public static String[] getFilesFromOptions(String[] args) {
	return null;
    }

    static class DeclReader {
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public void read(File f) { }
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public void primitive_declaration_types() { }
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public void write_decl(String decl) { }
    }

	static class File {
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public File(String s) {}
	}
}
