/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        try {
            if (true && true) {
            try {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 5; __cfwr_i41++) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 6; __cfwr_i88++) {
            Long __cfwr_item2 = null;
        }
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e48) {
            // ignore
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

    protected Character __cfwr_util670(Float __cfwr_p0, char __cfwr_p1) {
        return (null % -16L);
        try {
            while (false) {
            return 46.59f;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        for (int __cfwr_i21 = 0; __cfwr_i21 < 10; __cfwr_i21++) {
            boolean __cfwr_temp60 = true;
        }
        for (int __cfwr_i21 = 0; __cfwr_i21 < 1; __cfwr_i21++) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 7; __cfwr_i47++) {
            return null;
        }
        }
        return null;
    }
}