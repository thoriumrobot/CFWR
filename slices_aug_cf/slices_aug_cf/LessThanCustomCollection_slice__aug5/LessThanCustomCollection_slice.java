/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        for (int __cfwr_i62 = 0; __cfwr_i62 < 3; __cfwr_i62++) {
            try {
            byte __cfwr_result9 = null;
        } catch (Exception __cfwr_e70) {
         
        while (true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
   // ignore
        }
        }

    this(array, 0, array.length);
  }

  private LessThanCustomCollection(
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int start, @IndexOrHigh("#1") int end) {
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    this.end = end;
    // start is @LessThan(end + 1) but should be @LessThan(this.end + 1)
    // :: error: (assignment)
    this.start = start;
  }

  @Pure
  public @LengthOf("this") int length() {
    return end - start;
  }

  public double get(@IndexFor("this") int index) {
    // TODO: This is a bug.
    // :: error: (argument)
    checkElementIndex(index, length());
    // Because index is an index for "this" the index + start
    // must be an index for array.
    // :: error: (array.access.unsafe.high)
    return array[start + index];
  }

  public static @NonNegative int checkElementIndex(
      @LessThan("#2") @NonNegative int index, @NonNegative int size) {
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException();
    }
    return index;
  }

  public @IndexOrLow("this") int indexOf(double target) {
    for (int i = start; i < end; i++) {
      if (areEqual(array[i], target)) {
        // Don't know that it is greater than start.
        // :: error: (return)
        return i - start;
      }
    }
    return -1;
      protected static char __cfwr_process284(double __cfwr_p0, boolean __cfwr_p1) {
        try {
            while (false) {
            if (((null << null) ^ (-79.73 + -5)) && false) {
            return ((null - -82.44) >> true);
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        if (true || true) {
            while (true) {
            try {
            try {
            try {
            Float __cfwr_var45 = null;
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return 'x';
    }
    public int __cfwr_helper818(Float __cfwr_p0) {
        Float __cfwr_data59 = null;
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        if (false && false) {
            try {
            try {
            String __cfwr_item84 = "temp74";
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
        if ((true + false) || false) {
            while (false) {
            while (true) {
            try {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 1; __cfwr_i60++) {
            return (59.50 + (null << null));
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return 246;
    }
}
