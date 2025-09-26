/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        for (int __cfwr_i81 = 0; __cfwr_i81 < 2; __cfwr_i81++) {
            while (false) {
            Boolean __cfwr_result94 = null;
            break; // Prevent 
        try {
            try {
            if (true || true) {
            double __cfwr_node83 = 32.21;
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
infinite loops
        }
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
      byte __cfwr_proc582() {
        return null;
        return (null ^ false);
    }
    static byte __cfwr_aux71(char __cfwr_p0) {
        while (true) {
            if (('5' * 29.07f) || false) {
            if ((-973L ^ -467L) && false) {
            if ((true >> -895) || true) {
            if (false || true) {
            Object __cfwr_entry58 = null;
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        while (false) {
            if (true && (-798 ^ -362)) {
            return false;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i30 = 0; __cfwr_i30 < 8; __cfwr_i30++) {
            boolean __cfwr_entry82 = ((-845 - null) + true);
        }
        return (-26L ^ ('z' - -60.91));
        return (-307L % false);
    }
    private static long __cfwr_proc374(Character __cfwr_p0, char __cfwr_p1) {
        for (int __cfwr_i5 = 0; __cfwr_i5 < 7; __cfwr_i5++) {
            return null;
        }
        for (int __cfwr_i2 = 0; __cfwr_i2 < 9; __cfwr_i2++) {
            try {
            char __cfwr_entry6 = 'A';
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        }
        Long __cfwr_item26 = null;
        return 705L;
    }
}
