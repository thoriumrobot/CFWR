import org.checkerframework.checker.index.qual.LTLengthOf;

public class NotEnoughOffsets {

    void badParam2(@LTLengthOf(value = { "a" }, offset = { "c", "d" }) int x) {
    }
}
