public class ImmutableListOf {
    /**
     * Returns an immutable set containing the given elements, minus duplicates, in the order each was
     * first specified. That is, if multiple elements are {@linkplain Object#equals equal}, all except
     * the first are ignored.
     *
     * @since 3.0 (source-compatible since 2.0)
     */
        /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public static void of(Object e1, Object e2, Object e3, Object e4, Object e5, Object e6, Object... others) {
	final int paramCount = 6;
	Object[] elements = new Object[paramCount + others.length];
	// Note that all of these are found to be unsafe because the value checker can't prove the minlen of elements, because paramCount + others.length might overflow
	//:: error: (array.access.unsafe.high.constant)
	elements[0] = e1;
	//:: error: (array.access.unsafe.high.constant)
	elements[1] = e2;
	//:: error: (array.access.unsafe.high.constant)
	elements[2] = e3;
	//:: error: (array.access.unsafe.high.constant)
	elements[3] = e4;
	//:: error: (array.access.unsafe.high.constant)
	elements[4] = e5;
	//:: error: (array.access.unsafe.high.constant)
	elements[5] = e6;
   }
}
