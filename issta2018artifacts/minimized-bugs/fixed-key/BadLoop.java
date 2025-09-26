// corrected version of buggy Daikon code fixed in 5a599e7a1

public class BadLoop {
    // Outputs a sequence of space-separated characters, with (only) return
    // and newline quoted.  (Should backslash also be quoted?)
    /*@ public normal_behavior
      @ requires true;
      @ ensures true;
      @*/
    public static final void println_array_char_as_chars(
            PrintStream ps, Object[] a) {
        if (a == null) {
            ps.println("null");
            return;
        }
        ps.print('[');
        for (int i = 0; i < a.length; i++) {
            if (i != 0) ps.print(' ');
            char c = getCharValue(a[i]);
            if (c == '\r') {
                ps.print("\\r");
            } else if (c == '\n') { // not lineSep
                ps.print("\\n"); // not lineSep
            } else {
                ps.print(c);
            }
        }
        ps.println(']');
    }

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    private static char getCharValue(Object o) {
        return '!';
    }

    class PrintStream {

        /*@ public normal_behavior
    @ requires true;
    @ ensures true;
    @*/
        void println(char c) {
        }

        /*@ public normal_behavior
  @ requires true;
  @ ensures true;
  @*/
        void println(String s) {
        }

        /*@ public normal_behavior
        @ requires true;
        @ ensures true;
        @*/
        void print(char c) {
        }

        /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
        void print(String s) {
        }
    }
}
