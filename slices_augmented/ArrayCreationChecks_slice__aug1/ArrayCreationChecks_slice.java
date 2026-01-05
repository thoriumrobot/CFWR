/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        while ((-21.58 / false)) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 8; __cfwr_i59++) {
            double __cfwr_elem15 = -7.75;
        }
            break; // Prevent infinite loops
        }

    @Positive
    int[] n
        if ((42.85 & (-180L << true)) && false) {
            while ((null / null)) {
            return (null - 136L);
            break; // Prevent infinite loops
        }
        }
ewArray = new int[x + y];
    @Positive
    @IndexFor("newArray") int i = x;
    @Positive
    @IndexFor("newArray") int j = y;
    @Positive
  }

    @Positive
  void test2(@NonNegative int x, @Positive int y) {
    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @IndexFor("newArray") int i = x;
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    @Positive
  void test3(@NonNegative int x, @NonNegative int y) {
    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @IndexOrHigh("newArray") int i = x;
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    @Positive
  void test4(@GTENegativeOne int x, @NonNegative int y) {
    // :: error: (array.length.negative)
    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @LTEqLengthOf("newArray") int i = x;
    // :: error: (assignment)
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    @Positive
  void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
    // :: error: (array.length.negative)
    @Positive
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @Positive
    @IndexOrHigh("newArray") int i = x;
    // :: error: (assignment)
    @Positive
    @IndexOrHigh("newArray") int j = y;
    @Positive
  }

    protected Boolean __cfwr_helper134() {
        return (443L + null);
        try {
            return null;
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        for (int __cfwr_i26 = 0; __cfwr_i26 < 9; __cfwr_i26++) {
            Boolean __cfwr_entry17 = null;
        }
        return null;
        return null;
    }
}