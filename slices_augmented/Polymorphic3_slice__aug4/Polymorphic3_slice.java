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
        return (-85.24 ^ false);

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
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

    protected static Integer __cfwr_temp972(String __cfwr_p0, long __cfwr_p1, long __cfwr_p2) {
        return null;
        try {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 4; __cfwr_i8++) {
            long __cfwr_entry56 = -852L;
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        while ((37.26 ^ (null ^ 't'))) {
            return -439;
            break; // Prevent infinite loops
        }
        return null;
    }
    static Character __cfwr_func550(Double __cfwr_p0) {
        return null;
        Long __cfwr_item47 = null;
        return null;
    }
}