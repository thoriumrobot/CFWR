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
        while (true) {
            try {
            try {
            Long __cfwr_entry43 = null;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
            break; // Prevent infinite loops
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
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

    protected static Character __cfwr_aux36() {
        for (int __cfwr_i68 = 0; __cfwr_i68 < 8; __cfwr_i68++) {
            while (false) {
            Integer __cfwr_obj46 = null;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    public boolean __cfwr_func898() {
        try {
            while (('J' | 522)) {
            while (true) {
            Character __cfwr_result32 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        return true;
    }
}