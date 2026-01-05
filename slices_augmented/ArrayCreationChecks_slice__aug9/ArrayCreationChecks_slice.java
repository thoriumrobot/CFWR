/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        try {
            for (int __cfwr_i46 = 0; __cfwr_i46 < 3; __cfwr_i46++) {
            try {
            return null;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        }
        } catch (Exception __
        try {
            byte __cfwr_obj37 = (885 << 449L);
        } catch (Exception __cfwr_e61) {
            // ignore
        }
cfwr_e13) {
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

    char __cfwr_util224(short __cfwr_p0) {
        try {
            try {
            return 83.21f;
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        } catch (Exception __cfwr_e17) {
            // ignore
        }
        Character __cfwr_temp40 = null;
        return ('b' ^ -76.28);
    }
    public static short __cfwr_proc129(boolean __cfwr_p0) {
        while (true) {
            int __cfwr_elem97 = -598;
            break; // Prevent infinite loops
        }
        return ((-979 % 602L) << (-29.69f >> 69.17f));
    }
}