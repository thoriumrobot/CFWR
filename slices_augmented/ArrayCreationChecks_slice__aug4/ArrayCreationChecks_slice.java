/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        while (true) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 3; __cfwr_i34++) {
            while ((null / (null << -11.68f))) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 1; __cfwr_i14++) {
            double __cfwr_entry94 = ((433 + null) ^ null);
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }

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

    protected static Object __cfwr_proc198() {
        try {
            long __cfwr_entry85 = -556L;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        return null;
    }
    private static boolean __cfwr_calc673(float __cfwr_p0) {
        for (int __cfwr_i38 = 0; __cfwr_i38 < 8; __cfwr_i38++) {
            return 84.54;
        }
        for (int __cfwr_i99 = 0; __cfwr_i99 < 7; __cfwr_i99++) {
            return null;
        }
        return false;
    }
}