// Simplified test case for Daikon commit 29dfd8dde

public class GetSimplifyFreeIndices {

	/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
	public static Object get_simplify_free_indices(Object... vars) {
	if (vars.length == 1) {
	    return vars[0];
	} else if (vars.length == 2) {
	    //:: error: array.access.unsafe.high.constant
	    return compose(vars[0], vars[2]);
	} else {
		return null;
	}
    }

	/*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public static Object compose(Object obj1, Object obj2) {
	return null;
    }
}
