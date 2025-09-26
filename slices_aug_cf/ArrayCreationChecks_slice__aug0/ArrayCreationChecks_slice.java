/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
  void test1(@Positive int x, @Positive int y) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }

    int[] newArray = new int[x + y];
    @IndexFor("newArray")
        if (true && true) {
            return true;
        }
 int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
    int[] newArray = new int[x + y];
    @IndexOrHigh("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test4(@GTENegativeOne int x, @NonNegative int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    @LTEqLengthOf("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

  void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @IndexOrHigh("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

  void test6(int x, int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @IndexFor("newArray") int i = x;
    // :: error: (assignment)
    @IndexOrHigh("newArray") int j = y;
  }

    private short __cfwr_util891(String __cfwr_p0, Float __cfwr_p1) {
        return null;
        return null;
    }
    private static int __cfwr_compute569(Character __cfwr_p0, byte __cfwr_p1, String __cfwr_p2) {
        if (true && true) {
            for (int __cfwr_i2 = 0; __cfwr_i2 < 4; __cfwr_i2++) {
            if (false && true) {
            Long __cfwr_var13 = null;
        }
        }
        }
        for (int __cfwr_i60 = 0; __cfwr_i60 < 8; __cfwr_i60++) {
            Object __cfwr_temp69 = null;
        }
        for (int __cfwr_i78 = 0; __cfwr_i78 < 10; __cfwr_i78++) {
            if (true && true) {
            while ((null << 9.41)) {
            while ((18.93f % false)) {
            if (((null >> null) - (98.89f << 's')) && (915L / 614)) {
            while (false) {
            char __cfwr_val27 = '5';
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        Double __cfwr_val97 = null;
        return -870;
    }
}