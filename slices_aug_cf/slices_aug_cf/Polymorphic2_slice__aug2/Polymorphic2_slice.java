/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testUpperBound2(@LTLengthOf("array1") int a, @LTEqLengthOf("array1") int b) {
        Long __cfwr_var16 = null;

    @LTEqLengthOf("array1") int z = mergeUpperBound(a, b);
    // :: error: (assignment)
    @LTLengthOf("array1") int zz = mergeUpperBound(a, b);
      pub
        Long __cfwr_var33 = null;
lic static Object __cfwr_aux691() {
        try {
            while (true) {
            boolean __cfwr_node6 = false;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        return null;
    }
    static Long __cfwr_helper957(Boolean __cfwr_p0) {
        while (false) {
            return "temp84";
            break; // Prevent infinite loops
        }
        return null;
    }
}
