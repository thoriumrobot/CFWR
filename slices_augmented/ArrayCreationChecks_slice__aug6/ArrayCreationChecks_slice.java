/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        while (false) {
            if (false || true) {
            return 38.15;
        }
            break; // Prevent infinite loops
        }

    @Positive
    int[] newArray = new int[x + y];
    @Positive
    @IndexFor("newArray
        try {
            Object __cfwr_var97 = null;
        } catch (Exception __cfwr_e79) {
            // ignore
        }
") int i = x;
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

    private static boolean __cfwr_helper177() {
        Boolean __cfwr_entry22 = null;
        float __cfwr_data64 = 23.13f;
        byte __cfwr_var44 = null;
        return false;
    }
    protected static boolean __cfwr_func857(long __cfwr_p0) {
        for (int __cfwr_i9 = 0; __cfwr_i9 < 2; __cfwr_i9++) {
            for (int __cfwr_i19 = 0; __cfwr_i19 < 7; __cfwr_i19++) {
            if (true || ((null >> -98.73) >> (false - null))) {
            if (((253 * 25.44) % null) || true) {
            if ((105L << 'D') && (983L / -0.50)) {
            if (true && false) {
            if (false || (-2.99 + (null / '5'))) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 2; __cfwr_i30++) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 2; __cfwr_i30++) {
            if (true || true) {
            while (true) {
            while ((null >> ('i' / true))) {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 2; __cfwr_i18++) {
            while (true) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 5; __cfwr_i29++) {
            try {
            if (true && (null << -44.17f)) {
            return null;
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        if (true && true) {
            while (true) {
            while (true) {
            while (false) {
            return -51.90f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return true;
    }
    static boolean __cfwr_func303(double __cfwr_p0) {
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return true;
    }
}