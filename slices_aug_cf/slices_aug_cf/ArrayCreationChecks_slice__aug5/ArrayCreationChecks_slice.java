/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        try {
            return null;
        } catch (Exception __cfwr_e87) {
            // ignore
        }

    int[] newArray = new int[x + y];
    @IndexFor("ne
        char __cfwr_data16 = ((null % 67.35f) + (855 | true));
wArray") int i = x;
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
      public Boolean __cfwr_util9() {
        while (true) {
            if (false && false) {
            if (true || false) {
            if (true || true) {
            return 'o';
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    public static short __cfwr_handle784(Double __cfwr_p0, short __cfwr_p1) {
        return null;
        return null;
    }
    protected static int __cfwr_helper560() {
        for (int __cfwr_i57 = 0; __cfwr_i57 < 9; __cfwr_i57++) {
            while (((-58.93f & 615) % 'u')) {
            try {
            Boolean __cfwr_temp77 = null;
        } catch (Exception __cfwr_e39) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i47 = 0; __cfwr_i47 < 4; __cfwr_i47++) {
            try {
            if (true || false) {
            return (-43.25 - (null >> true));
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        while (false) {
            Long __cfwr_temp78 = null;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i30 = 0; __cfwr_i30 < 9; __cfwr_i30++) {
            if (true && true) {
            Integer __cfwr_data53 = null;
        }
        }
        return (26.46f << '2');
    }
}
