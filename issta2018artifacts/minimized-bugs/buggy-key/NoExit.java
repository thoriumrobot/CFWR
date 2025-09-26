// code that causes the bug fixed in plume-lib by rev. 95b3cab

public class NoExit {

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
     public static void main(String[] args) {
        if (args.length != 2) {
          println("Needs 2 arguments, got " + args.length);
        }
	//:: error: (array.access.unsafe.high.constant)
        String limit = args[0];
     }

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    private static void println(String s) {
    }
}
