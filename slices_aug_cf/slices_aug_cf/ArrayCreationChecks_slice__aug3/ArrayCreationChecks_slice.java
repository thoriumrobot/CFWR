/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        Double __cfwr_item87 = null;

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
      protected static String __cfwr_util539() {
        while (((-413 | null) + -75.30f)) {
            return (null & null);
            break; // Prevent infinite loops
        }
        return "temp78";
    }
    private static Long __cfwr_process751() {
        Float __cfwr_obj54 = null;
        while (false) {
            if (false && true) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return null;
        for (int __cfwr_i33 = 0; __cfwr_i33 < 5; __cfwr_i33++) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 9; __cfwr_i6++) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 6; __cfwr_i69++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        return null;
    }
}
