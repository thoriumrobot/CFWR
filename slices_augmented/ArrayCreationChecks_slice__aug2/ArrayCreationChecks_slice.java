/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayCreationChecks_slice {
    @Positive
  void test1(@Positive int x, @Positive int y) {
        while (((95.00 + null) * (null | null))) {
            while (false) {
            while (true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // P
        for (int __cfwr_i53 = 0; __cfwr_i53 < 9; __cfwr_i53++) {
            try {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 1; __cfwr_i5++) {
            try {
            while (true) {
            try {
            while (true) {
            if (false || true) {
            if (true || false) {
            try {
            Character __cfwr_elem66 = null;
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        }
revent infinite loops
        }
            break; // Prevent infinite loops
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

    static Object __cfwr_calc72() {
        for (int __cfwr_i64 = 0; __cfwr_i64 < 1; __cfwr_i64++) {
            byte __cfwr_item75 = null;
        }
        return 353;
        if (false || false) {
            if ((false & false) && false) {
            if (true || true) {
            Long __cfwr_data37 = null;
        }
        }
        }
        if (false || false) {
            while (true) {
            return 'm';
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}