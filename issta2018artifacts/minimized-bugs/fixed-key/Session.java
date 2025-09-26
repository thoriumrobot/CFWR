// simplified version of bug fixed in Daikon commit cc9edee13

public class Session {
    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public static void do_session(InputStream is, byte[] buf) {
	int pos = is.read(buf);
        if (pos != -1) {
            //:: error: argument.type.incompatible
            String actual = new String(buf, 0, pos);
        }
    }

    private static class InputStream {
        /*@ public normal_behavior
@ requires true;
@ ensures \result >= -1;
@*/
        public int read(byte[] buf) {
            return -1;
        }
    }
}
