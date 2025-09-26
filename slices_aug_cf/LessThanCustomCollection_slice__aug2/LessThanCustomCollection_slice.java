/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private LessThanCustomCollection(int[] array) {
        if (true && true) {
            try {
            return null;
        } catch (Exception __cfwr_e3) {
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
      static Double __cfwr_calc251(Character __cfwr_p0) {
        for (int __cfwr_i7 = 0; __cfwr_i7 < 10; __cfwr_i7++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i81 = 0; __cfwr_i81 < 8; __cfwr_i81++) {
            return null;
        }
        while (true) {
            return -3.44f;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i49 = 0; __cfwr_i49 < 3; __cfwr_i49++) {
            try {
            Float __cfwr_result58 = null;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        }
        return null;
    }
    private Double __cfwr_helper751(int __cfwr_p0, String __cfwr_p1) {
        while (false) {
            return false;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i76 = 0; __cfwr_i76 < 6; __cfwr_i76++) {
            String __cfwr_data26 = "value74";
        }
        return null;
    }
    private Character __cfwr_helper344(Float __cfwr_p0) {
        for (int __cfwr_i3 = 0; __cfwr_i3 < 8; __cfwr_i3++) {
            Boolean __cfwr_result40 = null;
        }
        try {
            float __cfwr_var46 = (null >> 47.64f);
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        while ((null % 736)) {
            double __cfwr_item62 = 50.69;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i75 = 0; __cfwr_i75 < 4; __cfwr_i75++) {
            if (false && false) {
            if (true && false) {
            if ((true - 119) || (-21.85 << null)) {
            try {
            if (false && false) {
            Long __cfwr_entry43 = null;
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        }
        }
        }
        return null;
    }
}
