/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        if (true || false) {
            return null;
        }

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
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
      public static byte __cfwr_aux303() {
        for (int __cfwr_i3 = 0; __cfwr_i3 < 5; __cfwr_i3++) {
            if (false && true) {
            while (((true - null) + -73.78f)) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 1; __cfwr_i91++) {
            Double __cfwr_elem33 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        char __cfwr_item35 = 'J';
        while (((null - false) >> false)) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 8; __cfwr_i42++) {
            if (false || ((null & null) - 'O')) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        Double __cfwr_obj79 = null;
        return null;
    }
    protected Double __cfwr_proc792() {
        char __cfwr_obj66 = '4';
        while ((null >> (96.47f % null))) {
            while (false) {
            try {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            return false;
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
