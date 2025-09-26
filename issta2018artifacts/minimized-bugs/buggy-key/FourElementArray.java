// A simplified version of the bug fixed by Daikon commit d9e8fa8


public class FourElementArray {

    /* Every null in the function below was originally some meaningful String.
     * The original documentation specified that this function always returns
     * an array with four elements.
     */
        /*@ public normal_behavior
@ requires true;
@ ensures \result.length == 4;
@*/
    public static String[] esc_quantify(Object... vars) {
	if (vars.length == 1) {
	    //:: error: return.type.incompatible
	    return new String[] {null, null, ")"};
	} else if ((vars.length == 2) && boolean_condition(vars[1])) {
	    return new String[] {
		null,
		null,
		null,
		")"
	    };
	} else {
	    return new String[] {
		null, null, null, ")"
	    };
	}
    }

    /* In the actual example this was something meaningful. */
        /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    private static boolean boolean_condition(Object o) {
	return true;
    }
}
