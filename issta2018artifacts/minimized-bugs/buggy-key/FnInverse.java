// simplified version of the plume-lib bug fixed in 0abfea178a813c3895fecebcbb8bcbed1cee1b42

public class FnInverse {

	/*@ public normal_behavior
@ requires arange >= 0;
@ ensures true;
@*/
    public static int[] fn_inverse(int[] a, int arange) {
	int[] result = new int[arange];
	for (int i = 0; i < a.length; i++) {
	    int ai = a[i];
	    if (ai != -1) {
		//:: error: array.access.unsafe.high error: array.access.unsafe.low
		if (result[ai] != -1) {
			return null;
		}
		//:: error: array.access.unsafe.high error: array.access.unsafe.low
		result[ai] = i;
	    }
	}
	return result;
    }
}
