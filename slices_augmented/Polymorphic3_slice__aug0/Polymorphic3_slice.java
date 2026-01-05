/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic3_slice {
    @Positive
  void ubc_id(
    @Positive
      int[] a,
    @Positive
      int[] b,
    @Positive
      @LTLengthOf("#1") int ai,
    @Positive
      @LTEqLengthOf("#1") int al,
    @Positive
      @LTLengthOf({"#1", "#2"}) int abi,
    @Positive
      @LTEqLengthOf({"#1", "#2"}) int abl) {
        try {
            return null;
        } catch (Exception __cfwr_e74) {
            // ignore
        }

    @Positive
    int[] c;

    @Positive
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int 
        for (int __cfwr_i24 = 0; __cfwr_i24 < 5; __cfwr_i24++) {
            while ((('h' ^ 26.10) >> (null - 58.38f))) {
            byte __cfwr_var2 = null;
            break; // Prevent infinite loops
        }
        }
abl2 = identity(abl);
    @Positive
  }

    protected Object __cfwr_util248(Boolean __cfwr_p0, byte __cfwr_p1) {
        Float __cfwr_data29 = null;
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        if (true || (('A' | 84.30f) / (null ^ 80.00f))) {
            try {
            return null;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
        return null;
    }
}