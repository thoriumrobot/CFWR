/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test1(@Positive int x, @Positive int y) {
        try {
            Long __cfwr_node93 = null;
        } catch (Exception __cfwr_e66) {
            // ignore
        }

    int[] newArray = new int[x + y];
   
        short __cfwr_obj69 = (69.45 ^ (580 % 21.18));
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
      public static Long __cfwr_process112(Double __cfwr_p0) {
        try {
            if (true || (-698 << (false + false))) {
            while (false) {
            try {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 4; __cfwr_i93++) {
            try {
            float __cfwr_data39 = ((57.76f | -72.79) - ('l' - 'K'));
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        try {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 3; __cfwr_i30++) {
            try {
            if ((('8' ^ null) >> ('G' >> 80.19f)) && false) {
            while (true) {
            while (((null % null) + -414)) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 8; __cfwr_i22++) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 10; __cfwr_i98++) {
            return false;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        return null;
    }
    protected static short __cfwr_proc544(long __cfwr_p0, Float __cfwr_p1) {
        for (int __cfwr_i67 = 0; __cfwr_i67 < 6; __cfwr_i67++) {
            if (false && false) {
            while (false) {
            try {
            try {
            if (true || ((-3.23 + 'A') >> (462L | -96.62f))) {
            double __cfwr_result14 = -47.76;
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        boolean __cfwr_elem61 = ((-140 >> 34.91f) % null);
        return null;
        return null;
    }
}
