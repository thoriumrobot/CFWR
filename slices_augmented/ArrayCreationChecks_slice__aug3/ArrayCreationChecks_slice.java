/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        Float __cfwr_data68 = null;

    @Positive
    int[] newArray = new int[x + y];
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

    protected char __cfwr_helper712(String __cfwr_p0, long __cfwr_p1) {
        try {
            if (false && (-44.43 - (true % true))) {
            return null;
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        for (int __cfwr_i90 = 0; __cfwr_i90 < 4; __cfwr_i90++) {
            while ((981 << (null * null))) {
            while (true) {
            if (false && true) {
            char __cfwr_elem52 = 'E';
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
        return (-81 | false);
    }
    private static Long __cfwr_calc626(Integer __cfwr_p0, String __cfwr_p1) {
        String __cfwr_entry51 = "data94";
        return null;
        return 'a';
        return null;
    }
}