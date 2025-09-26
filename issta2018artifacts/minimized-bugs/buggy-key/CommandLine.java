// Simplified test case for bug fixed by Daikon commit f123f498b

public class CommandLine {

    private static boolean dump_dtrace;
    
    /** Reads a decl file and dumps statistics. */
            /*@ public normal_behavior
  @ requires true;
  @ ensures true;
  @*/
    public static void main(String[] args) {
	String[] files = getFilesFromOptions(args);
     
	// If reading/dumping dtrace file, just read one file and dump it
	if (dump_dtrace) {
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
                /*@ public normal_behavior
  @ requires true;
  @ ensures true;
  @*/
    public static String[] getFilesFromOptions(String[] args) {
	return null;
    }

    static class DTraceReader {
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public void read(File f) { }
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public void dump_data() { }
    }

    static class File {
		/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
		public File(String s) {}
	}
}
